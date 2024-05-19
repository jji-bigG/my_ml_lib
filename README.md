# My ML Project

This project is a learning exercise following my course at Cornell CS 4780 during my freshman year. It aims to consolidate the various machine learning and deep learning algorithms we developed in our Vocareum projects. Additionally, it serves as a reference for my actual machine learning library project, which is being developed in Rust.

## Project Overview

This repository implements a range of machine learning and deep learning techniques using only NumPy for computations. The primary objective is to understand and implement these algorithms from scratch, reinforcing the concepts learned during the course.

## Machine Learning Techniques

### Kernel Ensemble

### Traditional Machine Learning Algorithms

These algorithms are implemented using a custom linear algebra module. The common ones are:

- **Bagging:** Allows you to supply your own algorithm and set the bagging parameters.
- **Boosting**
- **Decision Trees:** Including variations like Random Forests and XGBoost.
- **Support Vector Machines**
- **Perceptron**

**Note:** Common kernel functions are pre-implemented, but you can supply your own by providing an anonymous function.

### Deep Learning

#### Neural Networks

Inspired by PyTorch, many ideas are borrowed from it, including:

- `autograd()`: Automatically differentiates during the forward pass.
- `Optimizer`: Various optimization algorithms.

With the modular definitions, the two other most popular deep learning architectures are also implemented:

- **Convolutional Neural Networks (CNNs)**
- **Transformers**

Specifically, you can use:

- Self Attention; multi-headed attention
- Convolution layers

Along with their feature engineering helpers.

**Common activation functions** are implemented, but you can define your own by following the documentation and checking our implementations.

### Reinforcement Learning

Support for reinforcement learning will be added after completing CS 4789, Cornell’s Reinforcement Learning course, and understanding the implementations of common libraries that support reinforcement learning.

## Project Structure

```bash
my_ml_project/
│
├── data/
│   ├── __init__.py
│   └── dataset.py
│
├── models/
│   ├── __init__.py
│   └── linear_regression.py
│
├── utils/
│   ├── __init__.py
│   └── data_preprocessing.py
│
├── tests/
│   ├── __init__.py
│   └── test_linear_regression.py
│
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
