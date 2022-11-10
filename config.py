import os
import numpy as np
import torch
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
# Loss weight of team1_win and detail_result predictions
TEAM1_WIN_LOSS_WEIGHT = 0.7
INFO_LOSS_WEIGHT = 1 - TEAM1_WIN_LOSS_WEIGHT
# Loss weight of each columns of detail_result
INFO_COLS_LOSS_WEIGHT = [5] + [0.1] * 13
INFO_COLS_LOSS_WEIGHT = np.array(INFO_COLS_LOSS_WEIGHT) / np.sum(INFO_COLS_LOSS_WEIGHT)
INFO_COLS_LOSS_WEIGHT = torch.tensor(INFO_COLS_LOSS_WEIGHT).to(device)
INFO_COLS = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
WIN_INFO_COLS = ['W' + col for col in INFO_COLS]
LOSE_INFO_COLS = ['L' + col for col in INFO_COLS]

# file
GENDER = 'M'
DATA_DIR = f'./data/MDataFiles_Stage2'

REGULAR_FILE = f'{GENDER}RegularSeasonDetailedResults.csv'
NCAA_FILE = f'{GENDER}NCAATourneyDetailedResults.csv'
SAMPLE_SUBMISSION_FILE = f'{GENDER}SampleSubmissionStage2.csv'

# Season
if GENDER == 'M':
    SEASONS = list(range(2003, 2020)) + [2021, 2022]
else:
    SEASONS = list(range(2010, 2020)) + [2021, 2022]