# coding: utf-8

import numpy as np
import torch

__all__ = ["irt3pl", "irt2pl", "irt1pl"]  # Update the exported function names


def irt3pl(theta, a, b, c, D=1.702):
    return c + (1 - c) / (1 + torch.exp(-D * a * (theta - b)+ 1e-8))

# def irt3pl(theta, a, b, c, D=1.702, *, F=np):
#     return c + (1 - c) * F.exp(D * a * (theta - b)) / (1 + F.exp(D * a * (theta - b)))


def irt2pl(theta, a, b, D=1.702, *, F=np):
    """Two Parameter Item Response Theory Model."""
    return F.exp(D * a * (theta - b)) / (1 + F.exp(D * a * (theta - b)))
#               1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))

# def irt2pl(theta, a, b, D=1.702, *, F=np):
#     """Two Parameter Item Response Theory Model."""
#     return F.exp(a * (theta - b)) / (1 + F.exp(a * (theta - b)))

def irt1pl(theta, a, b, c, D=1.702, *, F=np):
    """One Parameter Item Response Theory Model (Rasch Model)."""
    return F.exp(theta - b) / (1 + F.exp(theta - b))