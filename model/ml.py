import torch
import torch.nn as nn


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
