import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from easydict import EasyDict as edict
import config


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
        qt = QuantileTransformer(random_state=config.SEED)
        info_data = np.concatenate((df[config.WIN_INFO_COLS].values, df[config.LOSE_INFO_COLS].values), axis=0)
        qt.fit(info_data)
        return qt

    def process_df(self, _df, is_train=True):
        df = _df.copy()
        df.drop(columns=['WLoc', 'NumOT'], inplace=True)

        # normalize
        df[config.WIN_INFO_COLS] = self.normalizer.transform(df[config.WIN_INFO_COLS])
        df[config.LOSE_INFO_COLS] = self.normalizer.transform(df[config.LOSE_INFO_COLS])

        # map indices
        df['WTeamID'] = df['WTeamID'].astype(int).map(self.team_id_map)
        df['LTeamID'] = df['LTeamID'].astype(int).map(self.team_id_map)

        ret = []
        for _, group in df.groupby(['Season', 'DayNum']):
            data1 = group[['WTeamID'] + config.WIN_INFO_COLS].values
            data2 = group[['LTeamID'] + config.LOSE_INFO_COLS].values

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

def get_df(file_name):
    return pd.read_csv(f'{config.DATA_DIR}/{file_name}')

