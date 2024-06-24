import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a positive integer but got {value}")
    return ivalue


def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a non-negative integer but got {value}")
    return ivalue


def float_0_1(value):
    fvalue = float(value)
    if not (0 <= fvalue <= 1):
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a float from range (0, 1), but got {value}")
    return fvalue


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected!")


class ArgParser(ArgumentParser):
    def arg(self, *args, **kwargs):
        return super().add_argument(*args, **kwargs)

    def flag(self, *args, **kwargs):
        return super().add_argument(*args, action="store_true", **kwargs)

    def boolean_flag(self, *args, **kwargs):
        return super().add_argument(*args, type=str2bool, nargs="?", const=True, metavar="BOOLEAN", **kwargs)


def get_main_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Runtime
    p.arg("--exec-mode",
          type=str,
          default="train",
          choices=["all", "analyze", "preprocess", "train", "test", "eval"],
          help="Run all of the MIST pipeline or an individual component"),
    p.arg("--loss",
          type=str,
          default="dice_ce",
          choices=["dice_ce", "dice", "gdl", "gdl_ce", "bl", "hl", "gsl","myloss"],
          help="Loss function for training")
    p.arg("--model",
          type=str,
          default="unet",
          choices=["nnunet", "unet", "resnet", "densenet","donet"])
    p.arg("--oversampling",
          type=float_0_1,
          default=0.3,
          help="Probability of crop centered on foreground voxel")
    p.arg("--folds", nargs="+", default=[0], type=int, help="Which folds to run")
    p.arg("--epochs", type=positive_int, default=500, help="Number of epochs")
    p.arg("--numpy", type=str, default='numpy/dtm_numpy', help="Path to save preprocessed numpy data")


    p.arg("--data", type=str, default='dataset/dataset.json',help="Path to dataset json file")
    p.arg("--config", type=str, default='config',help="Path to config json file")
    p.arg("--gpus", nargs="+", default=[-1], type=int, help="Which gpu(s) to use, defaults to all available GPUs")
    p.arg("--num-workers", type=positive_int, default=8, help="Number of workers to use for data loading")
    p.arg("--master-port", type=str, default="12356", help="Master port for multi-gpu training")
    p.arg("--seed", type=non_negative_int, default=42, help="Random seed")
    p.boolean_flag("--tta", default=True, help="Enable test time augmentation")

    # Output
    p.arg("--results", type=str, default='results', help="Path to output of MIST pipeline")
    

    # AMP
    p.boolean_flag("--amp", default=False, help="Enable automatic mixed precision (recommended)")

    # Training hyperparameters
    p.arg("--batch-size", type=positive_int, default=8, help="Batch size")
    p.arg("--patch-size", nargs="+", default=[96, 96, 96], type=int, help="Height, width, and depth of patch size to "
                                                                          "use for cropping")
    p.arg("--learning-rate", type=float, default=0.001, help="Learning rate")
    p.arg("--exp_decay", type=float, default=0.9, help="Exponential decay factor")
    p.arg("--lr-scheduler",
          type=str,
          default="constant",
          choices=["constant", "cosine_warm_restarts", "exponential"],
          help="Learning rate scheduler")
    p.arg("--cosine-first-steps",
          type=positive_int,
          default=500,
          help="Length of a cosine decay cycle in steps, only with cosine_annealing scheduler")

    # Optimizer
    p.arg("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw"], help="Optimizer")
    p.boolean_flag("--clip-norm", default=False, help="Use gradient clipping")
    p.arg("--clip-norm-max", type=float, default=1.0, help="Max threshold for global norm clipping")

    # Neural network parameters

    p.boolean_flag("--pocket", default=False, help="Use pocket version of network")
    p.arg("--depth", default=4, type=non_negative_int, help="Depth of U-Net or similar architecture")
    p.arg("--init-filters", type=non_negative_int, default=32, help="Number of filters to start network")
    p.boolean_flag("--deep-supervision", default=False, help="Use deep supervision")
    p.arg("--deep-supervision-heads", type=positive_int, default=2, help="Number of deep supervision heads")
    p.boolean_flag("--vae-reg", default=False, help="Use VAE regularization")
    p.arg("--vae-penalty", type=float_0_1, default=0.1, help="Weight for VAE regularization loss")
    p.boolean_flag("--l2-reg", default=False, help="Use L2 regularization")
    p.arg("--l2-penalty", type=float_0_1, default=0.00001, help="L2 penalty")
    p.boolean_flag("--l1-reg", default=False, help="Use L1 regularization")
    p.arg("--l1-penalty", type=float_0_1, default=0.00001, help="L1 penalty")

    # Data loading


    # Preprocessing
    p.boolean_flag("--use-n4-bias-correction", default=False, help="Use N4 bias field correction (only for MR images)")
    p.boolean_flag("--use-precomputed-class-weights", default=False, help="Use precomputed class weights")
    p.arg("--class-weights", nargs="+", type=float, help="Specify class weights")

    # Loss function

    p.arg("--alpha-scheduler",
          type=str,
          default="linear",
          choices=["linear", "step", "cosine"],
          help="Choice of alpha scheduler for boundary losses")
    p.arg("--linear-schedule-pause",
          type=positive_int,
          default=5,
          help="Number of epochs before linear alpha scheduler starts")
    p.arg("--step-schedule-step-length",
          type=positive_int,
          default=5,
          help="Number of epochs before in each section of the step-wise alpha scheduler")


    # Sliding window inference
    p.arg("--sw-overlap",
          type=float_0_1,
          default=0.25,
          help="Amount of overlap between scans during sliding window inference")
    p.arg("--blend-mode",
          type=str,
          choices=["constant", "gaussian"],
          default="constant",
          help="How to blend output of overlapping windows")

    # Postprocessing
    p.boolean_flag("--postprocess", default=True, help="Run post processing on MIST output")
    p.boolean_flag("--post-no-morph",
                   default=False,
                   help="Do not try morphological smoothing for postprocessing")
    p.boolean_flag("--post-no-largest",
                   default=False,
                   help="Do not run connected components analysis for postprocessing")

    # Validation
    p.arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")

    p.arg("--steps-per-epoch",
          type=positive_int,
          help="Steps per epoch. By default ceil(training_dataset_size / batch_size / gpus)")

    args = p.parse_args()
    args.results = os.path.join(args.results, args.model)
    args.results = os.path.join(args.results, args.loss)
    return args
