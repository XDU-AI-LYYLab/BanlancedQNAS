# Balanced-Quantum-Neural-Architecture-Search
This repository contains the code implementation for the paper "Banlanced Quantum Neural Architecture Search".
[BQNAS paper:] (https://doi.org/10.1016/j.neucom.2024.127860)

# Abstract
In the last decade, there has been continuously increasing attention on Neural Architecture Search (NAS). The design of network architecture is aimed at automatically generating efficient neural networks in the absence of prior knowledge. Most of current oneshot NAS methods are based on weight sharing. However, there are two main problems that lead to suboptimal search results. Firstly, the large scale supernet results in a complex and difficult search for optimal subnetwork structures in the large discrete search space. Secondly, weight sharing makes the internal weight parameters of the supernet cannot always vary towards the final weights required for the optimal network structure due to the influence of different structures. These problems are responsible for the fact that the performance of the subnetworks searched by the supernet cannot represent the real performance of the subnetworks trained from scratch. To solve these problems, we propose the method based on quantum evolution and balance pool, called BQNAS, which consists of two stages. The single path supernet is trained based on weight sharing and balance pool sampling method. Then, quantum parallelism is exploited for one-hot encoding and searching for the optimal subnetwork structure based on quantum evolutionary algorithms. It can greatly improve the search efficiency of oneshot approaches in subnetwork sampling and evaluate the real performance for searching for superior subnetwork structure. Extensive experiments on two datasets show that the proposed approach outperforms the state-of-the-art ones. Specifically, BQNAS achieves a top-1 accuracy of 97.27% on CIFAR-10 and 81.36% on CIFAR-100 datasets.

# Contribution
1) The quantum coding proposed by us can sufficiently reduce the search cost and can be used in quantum evolutionary algorithms.
2) We propose the quantum evolution algorithm to enhance the search efficiency of oneshot methods in huge search space, so that we can easily search for the optimal network architecture.
3) We propose the balance pool strategy to enhance the ability to find excellent subnetworks from the supernet.
4) Comparisons with a large number of manually designed networks and neural architecture search methods demonstrate that the proposed approach performs favorably against state-of-the-art methods.

# Datasets
CIFAR-10 and CIFAR-100

# Usage
pytorch 1.12

The configs can be adjusted in lib/config_file/

  Searching stage:
  python search_example.py 

  Training stage:
  python evaluate_example.py



# Citation
If you find this code implementation useful in your research, please consider citing the following paper:
@article{LI2024127860,
title = {Balanced quantum neural architecture search},
journal = {Neurocomputing},
pages = {127860},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.127860},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224006313},
author = {Yangyang Li and Guanlong Liu and Peixiang Zhao and Ronghua Shang and Licheng Jiao},
}
