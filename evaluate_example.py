from lib.config_file.arg_config import *
from lib.container import Evaluator
from thop import profile
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    args = get_init_config("evaluate")

    evaluator = Evaluator(args)
    evaluator.evaluate()
    evaluator.inference()
