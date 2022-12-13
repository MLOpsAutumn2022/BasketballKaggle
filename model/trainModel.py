import torch.nn as nn

from config import *
from data_preproccesing.data_preproccesing import get_df, DataManager
from model.ml import IterativeModel, MetricTracker


def train_model():
    sub_df = get_df(SAMPLE_SUBMISSION_FILE)
    reg_df = get_df(REGULAR_FILE)
    nca_df = get_df(NCAA_FILE)
    # Аналогично для SEED, WIN_INFO_COLS, LOSE_INFO_COLS. Хотелось, чтобы они были объявлены внутри функции
    dm = DataManager(reg_df, nca_df, sub_df, device)

    model = IterativeModel(
        num_team=dm.num_team,
        num_info=len(INFO_COLS),
        team_emb_size=TEAM_EMB_SIZE,
        model_emb_size=MODEL_HIDDEN_SIZE,
    ).to(device)

    bce = nn.BCELoss()
    mse = nn.MSELoss(reduction="none")

    test_pred = []
    valid_bce = []
    valid_mse = []

    for season in SEASONS:
        print(f" ======== Season {season} ======== ")

        # get data of given season
        train_data, valid_data, test_data = dm.get_train_data(season=season)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # train
        model.train()
        mt = MetricTracker()
        for data in train_data:
            # pred
            team1_win_pred, team1_pred, team2_pred = model(
                data.team1_ids, data.team2_ids
            )

            # compute loss
            bce_loss = bce(team1_win_pred, data.team1_win)
            team1_info_loss = mse(team1_pred, data.team1_data)
            team2_info_loss = mse(team2_pred, data.team2_data)
            mse_loss = torch.mean(
                (team1_info_loss + team2_info_loss) / 2 * INFO_COLS_LOSS_WEIGHT
            )
            loss = TEAM1_WIN_LOSS_WEIGHT * bce_loss + INFO_LOSS_WEIGHT * mse_loss

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update metric
            mt.update(count=len(team1_pred), bce=bce_loss.item(), mse=mse_loss.item())
        print(f"Train Season: {season}, {mt}")

        # valid
        if valid_data:
            model.eval()
            mt = MetricTracker()
            for data in valid_data:
                # pred
                team1_win_pred, team1_pred, team2_pred = model(
                    data.team1_ids, data.team2_ids
                )

                # compute loss
                bce_loss = bce(team1_win_pred, data.team1_win)
                team1_info_loss = mse(team1_pred, data.team1_data)
                team2_info_loss = mse(team2_pred, data.team2_data)
                mse_loss = torch.mean(
                    (team1_info_loss + team2_info_loss) / 2 * INFO_COLS_LOSS_WEIGHT
                )

                # update metric
                mt.update(
                    count=len(team1_pred), bce=bce_loss.item(), mse=mse_loss.item()
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

    return valid_bce, valid_mse
