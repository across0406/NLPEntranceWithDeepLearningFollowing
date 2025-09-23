# NLPEntranceWithDeepLearningFollowing
This repository is the following records for "Introduction to Natural Language Processing Using Deep Learning"

## References
- Introduction to Natural Language Processing Using Deep Learning (https://wikidocs.net/book/2155)

## Contents

1. [Environment](#1.-Environment)
2. [Installation](#2.-Installation)

## 1. Environment

| Item Name | Item Value |
| :--- | :--- |
| OS | Windows 11 (WSL2 Ubuntu 20.04) |
| Language | Python 3.9.13 (virtualenv) |
| Framework | Tensorflow, PyTorch |
| CUDA | 12.9 (on WSL2) |
| CUDNN | 8.9.7 (on WSL2) |

## 2. Installation

- Pre-requisites
    - Python Interpreter (with pip)
    - (If you want to use CUDA) CUDA 12.9
    - (If you want to use CUDA) CUDNN 8.9.7
1. Choose the requirements text file that you want to use
    - e.g. requirements_python3.9_cuda12.9_cudnn8.9.7.txt
2. Please type the command to the following:
```
pip install -r [requirements text file]
# e.g. pip install -r requirements_python3.9_cuda12.9_cudnn8.9.7.txt
```