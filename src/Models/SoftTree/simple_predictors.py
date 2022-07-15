"""Simple Predictors implementing the PredSubModel interface"""
import tensorflow as tf
from Models.SoftTree.model_interfaces import PredSubModel
from Models.SoftTree.layers import LayerIface

class LinearPred(PredSubModel):
    """Linear or offset-only predictions
    linear vs. offset determined by layer passed in"""

    def __init__(self, layer: LayerIface):
        self.layer = layer

    