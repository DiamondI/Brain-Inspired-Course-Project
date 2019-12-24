# Brain-Inspired-Course-Project

## Description

This repo is for brain inspired course project, with the file structure as follows:

```
.
├── README.md
└── brain-inspired-project
    ├── SGD
    │   ├── __pycache__
    │   │   ├── gradient_2d.cpython-36.pyc
    │   │   └── two_layer_net.cpython-36.pyc
    │   ├── gradient_1d.py
    │   ├── gradient_2d.py
    │   ├── gradient_method.py
    │   ├── gradient_simplenet.py
    │   ├── train_neuralnet.py
    │   ├── train_neuralnet_mlp.py
    │   └── two_layer_net.py
    ├── common
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-36.pyc
    │   │   ├── functions.cpython-36.pyc
    │   │   ├── gradient.cpython-36.pyc
    │   │   └── util.cpython-36.pyc
    │   ├── functions.py
    │   ├── gradient.py
    │   ├── layers.py
    │   ├── multi_layer_net.py
    │   ├── multi_layer_net_extend.py
    │   ├── optimizer.py
    │   ├── trainer.py
    │   └── util.py
    └── dataset
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-36.pyc
        │   └── mnist.cpython-36.pyc
        ├── lena.png
        ├── lena_gray.png
        ├── mnist.pkl
        ├── mnist.py
        ├── t10k-images-idx3-ubyte.gz
        ├── t10k-labels-idx1-ubyte.gz
        ├── train-images-idx3-ubyte.gz
        └── train-labels-idx1-ubyte.gz

7 directories, 34 files
```

This finishes three topics:

- Topic A: How does the sample size influence the smoothness of the loss function.
- Topic B: How does the depth of network influence the smoothness of the loss function.
- Topic C: How does the number of iterations influence the smoothness of the loss function.

##  Dataset

The dataset is just MNIST, and it is included in the `dataset` directory.

## Main Program

The main program is `SGD/train_neuralnet_mlp.py`. For Topic B and Topic C, please modify the parameters in main program. However, for Topic A, which relates to sample input size, please modify the parameters in `dataset/mnist.py`.

## Dependencies

- matplotlib         2.2.2
- numpy               1.14.1
- sklearn               0.22