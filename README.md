# Score-Based Generative Models for Financial Time Series and Portfolio Generation

## Project Description

This project explores **Score-Based Generative Models (SGMs)** and their application to financial time series modelling and portfolio generation. The work combines stochastic calculus, machine learning, and quantitative finance to investigate whether diffusion-based generative models can be used to simulate realistic financial returns and improve portfolio allocation strategies.

The project includes both theoretical validation on synthetic distributions and practical experiments on real financial market data.

## Motivation

Financial markets exhibit:

- Heavy tails
- Volatility clustering
- Non-Gaussian behaviour
- Regime changes

Traditional models often fail to capture these properties. Diffusion generative models provide a promising alternative by learning the full data distribution instead of assuming parametric models.

## Project Goals

The main objectives of this project:

- Implement Score-Based Generative Models from scratch
- Validate convergence on known probability distributions
- Train neural networks to approximate score functions
- Generate synthetic financial return distributions
- Construct portfolios using generated data
- Compare SGMs with Schrödinger Bridge models

## Methodology

The project follows the standard diffusion model pipeline:

### 1 Forward diffusion

Data is gradually transformed into Gaussian noise using an Ornstein-Uhlenbeck process.

### 2 Score estimation

The score function is approximated using neural networks trained with:

- Score matching
- Stochastic Gradient Langevin Dynamics (SGLD)

### 3 Reverse process

Data is reconstructed from noise using reverse SDE simulation.

### 4 Portfolio optimization

Generated financial returns are used to optimize portfolio allocation via Sharpe ratio maximization.

## Experiments

### Synthetic distributions

The model was tested on:

- 1D Gaussian distribution
- 2D Gaussian distribution
- Exponential distribution
- Gaussian mixture models
- Multimodal distributions

These experiments verify correctness of the implementation.

### Financial data

The model was trained on historical returns of:

- AAPL
- MSFT
- GOOG
- AMZN
- NFLX

Training period:

2010 – 2019

Testing period:

2020 – 2022

## Results

Key findings:

- SGMs accurately reproduce Gaussian distributions
- Model successfully learns multimodal distributions
- Financial return distributions are captured reasonably well
- Portfolio generation produced strong returns in stable markets
- Performance deteriorates during crisis regimes

## Technologies

Python

Main libraries:

- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- pytorch

## Mathematical Tools

- Stochastic Differential Equations
- Diffusion models
- Wasserstein distance
- Score matching
- Stochastic optimization
- Sharpe ratio optimization


## Key Skills Demonstrated

This project demonstrates knowledge in:

Machine Learning:

- Generative models
- Neural networks
- Diffusion models
- Model evaluation

Quantitative Finance:

- Portfolio optimization
- Risk modelling
- Return distributions
- Financial time series

Mathematics:

- Probability theory
- Stochastic processes
- Numerical methods
- Optimization

## Future Work

Possible improvements:

- Regime switching diffusion models
- Better tail modelling
- Larger neural architectures
- Transformer-based score models
- GAN comparison
- GARCH comparison

