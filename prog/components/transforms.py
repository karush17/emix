"""Stores tensor transformations."""

import torch as th


class Transform:
    """Transforms a given tensor to a desired view."""

    def transform(self, tensor):
        """Transforms a given tensor.
        
        Args:
            tensor: input tensor.
        """
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        """Observe the output log.
        
        Args:
            vshape_in: input tensor shape.
            dtype_in: input data type.
        """
        raise NotImplementedError


class OneHot(Transform):
    """Transforms a tensor to one-hot representation.
    
    Attributes:
        out_dim: dimension of the output tensor.
    """

    def __init__(self, out_dim):
        """Initialize the class.
        
        Args:
            out_dim: output dimension of the tensor.
        """
        self.out_dim = out_dim

    def transform(self, tensor):
        """Transform the input tensor.
        
        Args:
            tensor: input tensor.
        """
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        """Infer output logs.
        
        Args:
            vshape_in: input tensor shape.
            dtype_in: data type of the input tensor.
        """
        return (self.out_dim,), th.float32
