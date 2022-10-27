import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from easydict import EasyDict as edict
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

# Random Seed
SEED = 2626
torch.manual_seed(SEED)
np.random.seed(SEED)

# device
DEVICE_IDS = ""
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_IDS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# model
TEAM_EMB_SIZE = 16
MODEL_HIDDEN_SIZE = 32
BATCH_SIZE = 2000

# model training
LEARNING_RATE = 1e-2
## Loss weight of team1_win and detail_result predictions
TEAM1_WIN_LOSS_WEIGHT = 0.7
INFO_LOSS_WEIGHT = 1 - TEAM1_WIN_LOSS_WEIGHT
## Loss weight of each columns of detail_result
INFO_COLS_LOSS_WEIGHT = [5] + [0.1] * 13
INFO_COLS_LOSS_WEIGHT = np.array(INFO_COLS_LOSS_WEIGHT) / np.sum(INFO_COLS_LOSS_WEIGHT)
INFO_COLS_LOSS_WEIGHT = torch.tensor(INFO_COLS_LOSS_WEIGHT).to(device)
INFO_COLS = ['Score','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']
WIN_INFO_COLS = ['W'+col for col in INFO_COLS]
LOSE_INFO_COLS = ['L'+col for col in INFO_COLS]

# file
GENDER = 'M'
if GENDER == 'M':
    DATA_DIR = f'../input/mens-march-mania-2022/MDataFiles_Stage2'
else:
    DATA_DIR = f'../input/womens-march-mania-2022/WDataFiles_Stage2'

REGULAR_FILE = f'{GENDER}RegularSeasonDetailedResults.csv'
NCAA_FILE = f'{GENDER}NCAATourneyDetailedResults.csv'
SAMPLE_SUBMISSION_FILE = f'{GENDER}SampleSubmissionStage2.csv'

# Season
if GENDER == 'M':
    SEASONS = list(range(2003, 2020)) + [2021, 2022]
else:
    SEASONS = list(range(2010, 2020)) + [2021, 2022]

def get_df(file_name):
    return pd.read_csv(f'{DATA_DIR}/{file_name}')


class DataManager:
    def __init__(self, reg_df, nca_df, sub_df, device):
        self.reg_df = reg_df.copy()
        self.nca_df = nca_df.copy()
        self.sub_df = sub_df.copy()
        self.device = device
        self.team_id_map = self.get_team_id_map()
        self.num_team = len(self.team_id_map)
        self.normalizer = self.get_normalizer()

    def get_test_data(self, season):
        df = self.sub_df.copy()
        df['Season'] = df['ID'].apply(lambda x: int(x.split('_')[0]))
        df = df[df.Season == season]
        team1_ids = df['ID'].apply(lambda x: int(x.split('_')[1])).astype(int).map(self.team_id_map)
        team2_ids = df['ID'].apply(lambda x: int(x.split('_')[2])).astype(int).map(self.team_id_map)
        team1_ids = torch.tensor(team1_ids.values).long().to(self.device)
        team2_ids = torch.tensor(team2_ids.values).long().to(self.device)
        return team1_ids, team2_ids

    def get_team_id_map(self):
        df = self.reg_df
        team_ids = set(list(df.WTeamID.unique()) + list(df.LTeamID.unique()))
        return {team_id: i for i, team_id in enumerate(team_ids)}

    def get_normalizer(self):
        df = self.reg_df.copy()
        qt = QuantileTransformer(random_state=SEED)
        info_data = np.concatenate((df[WIN_INFO_COLS].values, df[LOSE_INFO_COLS].values), axis=0)
        qt.fit(info_data)
        return qt

    def process_df(self, _df, is_train=True):
        df = _df.copy()
        df.drop(columns=['WLoc', 'NumOT'], inplace=True)

        # normalize
        df[WIN_INFO_COLS] = self.normalizer.transform(df[WIN_INFO_COLS])
        df[LOSE_INFO_COLS] = self.normalizer.transform(df[LOSE_INFO_COLS])

        # map indices
        df['WTeamID'] = df['WTeamID'].astype(int).map(self.team_id_map)
        df['LTeamID'] = df['LTeamID'].astype(int).map(self.team_id_map)

        ret = []
        for _, group in df.groupby(['Season', 'DayNum']):
            data1 = group[['WTeamID'] + WIN_INFO_COLS].values
            data2 = group[['LTeamID'] + LOSE_INFO_COLS].values

            if is_train:
                # Duplicate the data and make it symetrical to get rid of winner and loser
                _data1 = np.zeros((len(data1) * 2, *data1.shape[1:]))
                _data1[::2] = data1.copy()
                _data1[1::2] = data2.copy()

                _data2 = np.zeros((len(data2) * 2, *data2.shape[1:]))
                _data2[::2] = data2.copy()
                _data2[1::2] = data1.copy()

                data1 = _data1
                data2 = _data2

            tmp = {
                'team1_ids': torch.tensor(data1[:, 0]).long().to(self.device),
                'team2_ids': torch.tensor(data2[:, 0]).long().to(self.device),
                'team1_data': torch.tensor(data1[:, 1:]).float().to(self.device),
                'team2_data': torch.tensor(data2[:, 1:]).float().to(self.device),
                'team1_win': torch.tensor(data1[:, 1] > data2[:, 1]).float().to(self.device)
            }
            ret.append(edict(tmp))
        return ret

    def get_train_data(self, season=2016):
        train_df = self.reg_df[self.reg_df.Season == season]
        train_df = self.process_df(train_df)
        if season < 2022:
            valid_df = self.nca_df[self.nca_df.Season == season]
            valid_df = self.process_df(valid_df, is_train=False)
            test_data = None
        else:
            valid_df = None
            test_data = self.get_test_data(season)
        return train_df, valid_df, test_data


class MetricTracker:
    def __init__(self):
        self.bce = 0
        self.mse = 0
        self.count = 0

    def update(self, count, bce, mse):
        self.count += count
        self.bce += bce * count
        self.mse += mse * count

    def __str__(self):
        return f"bce: {self.bce / self.count:.04f}, mse: {self.mse / self.count:.04f}"


class IterativeModel(nn.Module):
    def __init__(self, num_team, num_info=14, team_emb_size=32, model_emb_size=64):
        super().__init__()

        self.team_embs = nn.Embedding(num_team, team_emb_size)

        self.mlp = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(team_emb_size * 2, model_emb_size * 6)),
            nn.LeakyReLU(),

            nn.BatchNorm1d(model_emb_size * 6),
            nn.utils.weight_norm(nn.Linear(model_emb_size * 6, model_emb_size * 4)),
            nn.LeakyReLU(),

            nn.BatchNorm1d(model_emb_size * 4),
            nn.utils.weight_norm(nn.Linear(model_emb_size * 4, model_emb_size * 2)),
            nn.LeakyReLU(),

        )
        self.output_win = nn.Sequential(
            nn.Linear(model_emb_size * 2, 1),
            nn.Sigmoid()
        )
        self.output_team1_info = nn.Linear(model_emb_size * 2, num_info)
        self.output_team2_info = nn.Linear(model_emb_size * 2, num_info)

    def forward(self, team1_ids, team2_ids):
        team1_embs = self.team_embs(team1_ids)
        team2_embs = self.team_embs(team2_ids)

        embs = torch.cat((team1_embs, team2_embs), dim=1)
        embs = self.mlp(embs)

        team1_win = self.output_win(embs).view(-1)
        team1_info = self.output_team1_info(embs)
        team2_info = self.output_team2_info(embs)
        return team1_win, team1_info, team2_info

sub_df = get_df(SAMPLE_SUBMISSION_FILE)
reg_df = get_df(REGULAR_FILE)
nca_df = get_df(NCAA_FILE)

dm = DataManager(reg_df, nca_df, sub_df, device)

model = IterativeModel(
    num_team = dm.num_team,
    num_info = len(INFO_COLS),
    team_emb_size = TEAM_EMB_SIZE,
    model_emb_size = MODEL_HIDDEN_SIZE
).to(device)

bce = nn.BCELoss()
mse = nn.MSELoss(reduction='none')

test_pred = []
valid_bce = []
valid_mse = []

for season in SEASONS:
    print(f' ======== Season {season} ======== ')

    # get data of given season
    train_data, valid_data, test_data = dm.get_train_data(season=season)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train
    model.train()
    mt = MetricTracker()
    for data in train_data:
        # pred
        team1_win_pred, team1_pred, team2_pred = model(data.team1_ids, data.team2_ids)

        # compute loss
        bce_loss = bce(team1_win_pred, data.team1_win)
        team1_info_loss = mse(team1_pred, data.team1_data)
        team2_info_loss = mse(team2_pred, data.team2_data)
        mse_loss = torch.mean((team1_info_loss + team2_info_loss) / 2 * INFO_COLS_LOSS_WEIGHT)
        loss = TEAM1_WIN_LOSS_WEIGHT * bce_loss + INFO_LOSS_WEIGHT * mse_loss

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # update metric
        mt.update(
            count=len(team1_pred),
            bce=bce_loss.item(),
            mse=mse_loss.item()
        )
    print(f"Train Season: {season}, {mt}")

    # valid
    if valid_data:
        model.eval()
        mt = MetricTracker()
        for data in valid_data:
            # pred
            team1_win_pred, team1_pred, team2_pred = model(data.team1_ids, data.team2_ids)

            # compute loss
            bce_loss = bce(team1_win_pred, data.team1_win)
            team1_info_loss = mse(team1_pred, data.team1_data)
            team2_info_loss = mse(team2_pred, data.team2_data)
            mse_loss = torch.mean((team1_info_loss + team2_info_loss) / 2 * INFO_COLS_LOSS_WEIGHT)

            # update metric
            mt.update(
                count=len(team1_pred),
                bce=bce_loss.item(),
                mse=mse_loss.item()
            )

            valid_bce.append(mt.bce / mt.count)
            valid_mse.append(mt.mse / mt.count)

        print(f"Valid Season: {season}, {mt}")

    if test_data:
        # test
        model.eval()
        team1_ids, team2_ids = test_data
        team1_win_pred, _, _ = model(team1_ids, team2_ids)
        test_pred += team1_win_pred.tolist()
        print("Run Testing")

print(f"\nmean bce: {np.mean(valid_bce):4f}")
print(f"mean mse: {np.mean(valid_mse):4f}")