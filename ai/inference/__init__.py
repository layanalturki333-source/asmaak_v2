"""Inference utilities: load model and run prediction on landmark sequences."""

from .predictor import load_model, SignPredictor

__all__ = ["load_model", "SignPredictor"]
