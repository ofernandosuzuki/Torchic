# Torchic - A Deep Learning Library

<div align="center">
  <img src="Torchic.webp" alt="Torchic Pokémon" width="300"/>
</div>

Hey folks! This is a college project where I implemented a Neural Network library from scratch using only NumPy.

## Why "Torchic"?

It's a reference to PyTorch (that everyone uses) but with the cute Pokémon name.

## Installation

```bash
pip install -r requirements.txt
```

## Ready-to-use examples

There are two notebooks you can use as a base:

#### MNIST (digit classification)
Check out `mlp_mnist.ipynb` - the classic handwritten digits dataset. Good to start with!

#### Boston Housing (regression)
Take a look at `mlp_boston.ipynb` - here we predict house prices. Cool to see how regression works.

## Want to test with other datasets?

Use the examples above as a template! The structure is pretty simple:
1. Load your data
2. Normalize if needed 
3. Create the model with `torchic.py`
4. Train and see what happens

---

*Deep Learning 1 course project by Fernando Suzuki - Data Science and Artificial Intelligence, PUCRS*