import csv
import os

class LossLogger:
    def __init__(self, log_file="logs/training_loss_log.csv"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Buat file baru dengan header kalau belum ada
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "test_loss", "lr", "epoch_time"])

    def log(self, epoch, train_loss, test_loss, lr, epoch_time):
        """Catat train loss, test loss, lr, dan waktu ke file CSV"""
        with open(self.log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, test_loss, lr, epoch_time])
