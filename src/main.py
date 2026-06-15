"""
this module is meant to save a snapshot of the training code and start the training
"""
import os
import datetime
from train import start_all
from src.log_experiments import save_code
from src.dcn_config import get_output_paths

if __name__ == "__main__":
    time_st = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder = get_output_paths()
    save_code(os.path.join(folder, time_st, "code.zip"))
    start_all(time_st)
