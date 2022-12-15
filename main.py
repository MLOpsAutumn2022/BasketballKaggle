import numpy as np

from model import trainModel

valid_bce, valid_mse = trainModel.train_model()

print(f"\nmean bce: {np.mean(valid_bce):4f}")
print(f"mean mse: {np.mean(valid_mse):4f}")