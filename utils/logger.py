from torch.utils.tensorboard import SummaryWriter
import os

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
