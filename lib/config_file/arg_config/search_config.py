import os
import argparse


def get_search_config():
    parser = argparse.ArgumentParser(description="Searching configuration")
    # Search config
    parser.add_argument(
        "--directly-search",
        action="store_true",
        default=False,
        help="Supernet training or not. If not, than direct search the best architecture based on the weight of sueprnet.")
    parser.add_argument(
        "--search-strategy",
        type=str,
        default="evolution",
        help="The way to search the best architecture[evolution, random_search, differentiable, differentiable_gumbel]")

    parser.add_argument(
        "--target-hc",
        type=int,
        default=1000000000,
        help="Target hardware constraint")
    parser.add_argument(
        "--hc-weight",
        type=float,
        default=0.000005,
        help="The weight of hardware constraint objective")

    parser.add_argument(
        "--info-metric",
        type=str,
        default="param",
        help="HC objective for searching")

    # Random search
    parser.add_argument(
        "--random-iteration",
        type=int,
        default=1000,
        help="The network architectures sample num for random search")
    # Evolution algorithm
    parser.add_argument(
        "--generation-num",
        type=int,
        default=500,#800
        help="Generation num for evolution algorithm")
    parser.add_argument(
        "--population",
        type=int,
        default=20,
        help="Population size for evoluation algorithm")
    parser.add_argument(
        "--parent-num",
        type=int,
        default=10,
        help="Parent size for evolution algorithm")
    # Differentiable
    parser.add_argument(
        "--a-optimizer",
        type=str,
        default="sgd",
        help="Optimizer for supernet training")
    parser.add_argument("--a-lr", type=float, default=0.0045)
    parser.add_argument("--a-weight-decay", type=float, default=0.0003)
    parser.add_argument("--a-momentum", type=float, default=0.9)

    parser.add_argument("--a-decay-step", type=int)
    parser.add_argument("--a-decay-ratio", type=float)

    parser.add_argument("--a-alpha", type=float)
    parser.add_argument("--a-beta", type=float)

    # Supernet config
    parser.add_argument(
        "--search-space",
        type=str,
        default="proxylessnas",
        help="Search spcae in different paper [proxylessnas, fbnet_s, fbnet_l, spos]")
    parser.add_argument(
        "--sample-strategy",
        type=str,
        default="fair",
        help="Sampling strategy for training supernet [fair, uniform, differentiable, differentiable_gumbel]")

    # Supernet training config
    parser.add_argument("--epochs", type=int, default=800,
                        help="The epochs for supernet training")

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="Optimizer for supernet training")
    parser.add_argument("--lr", type=float, default=0.0045)
    parser.add_argument("--weight-decay", type=float, default=0.0004)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--decay-step", type=int,default=20)
    parser.add_argument("--decay-ratio", type=float,default=0.8)

    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float,default=0.9)

    parser.add_argument(
        "--bn-momentum",
        type=float,
        default=0.1,
        help="Momentum for the BN")
    parser.add_argument(
        "--bn-track-running-stats",
        type=int,
        default=1,
        help="Track running stats")

    return parser
