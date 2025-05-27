import inspect
import warnings
from collections import OrderedDict
from typing import Callable, Union

import torch.nn
import torch.nn as nn
from torch.nn.parameter import Parameter

class Lambda(torch.nn.Module):
    """A wrapper for functions from `torch` and methods of `torch.Tensor`.

    An important "feature" of this module is that it is intentionally limited:

    - Only the functions from the `torch` module and the methods of `torch.Tensor`
      are allowed.
    - The passed callable must accept a single `torch.Tensor`
      and return a single `torch.Tensor`.
    - The allowed keyword arguments must be of simple types (see the docstring).

    **Usage**

    >>> m = delu.nn.Lambda(torch.squeeze)
    >>> m(torch.randn(2, 1, 3, 1)).shape
    torch.Size([2, 3])
    >>> m = delu.nn.Lambda(torch.squeeze, dim=1)
    >>> m(torch.randn(2, 1, 3, 1)).shape
    torch.Size([2, 3, 1])
    >>> m = delu.nn.Lambda(torch.Tensor.abs_)
    >>> m(torch.tensor(-1.0))
    tensor(1.)

    Custom functions are not allowed
    (technically, they are **temporarily** allowed,
    but this functionality is deprecated and will be removed in future releases):

    >>> # xdoctest: +SKIP
    >>> m = delu.nn.Lambda(lambda x: torch.abs(x))
    Traceback (most recent call last):
        ...
    ValueError: fn must be a function from `torch` or a method of `torch.Tensor`, but ...

    Non-trivial keyword arguments are not allowed:

    >>> m = delu.nn.Lambda(torch.mul, other=torch.tensor(2.0))
    Traceback (most recent call last):
        ...
    ValueError: For kwargs, the allowed value types include: ...
    """  # noqa: E501

    def __init__(self, fn: Callable[..., torch.Tensor], /, **kwargs) -> None:
        """
        Args:
            fn: the callable.
            kwargs: the keyword arguments for ``fn``. The allowed values types include:
                None, bool, int, float, bytes, str
                and (nested) tuples of these simple types.
        """
        super().__init__()
        if not callable(fn) or (
            fn not in vars(torch).values()
            and (
                fn not in (member for _, member in inspect.getmembers(torch.Tensor))
                or inspect.ismethod(fn)  # Check if fn is a @classmethod
            )
        ):
            warnings.warn(
                'Passing custom functions to delu.nn.Lambda is deprecated'
                ' and will be removed in future releases.'
                ' Only functions from the `torch` module and methods of `torch.Tensor`'
                ' are allowed',
                DeprecationWarning,
            )
            # NOTE: in future releases, replace the above warning with this exception:
            # raise ValueError(
            #     'fn must be a function from `torch` or a method of `torch.Tensor`,'
            #     f' but this is not true for the passed {fn=}'
            # )

        def is_valid_value(x):
            return (
                x is None
                or isinstance(x, (bool, int, float, bytes, str))
                or (isinstance(x, tuple) and all(map(is_valid_value, x)))
            )

        for k, v in kwargs.items():
            if not is_valid_value(v):
                raise ValueError(
                    'For kwargs, the allowed value types include:'
                    ' None, bool, int, float, bytes, str and (nested) tuples containing'
                    ' values of these simple types. This is not true for the passed'
                    f' argument {k} with the value {v}'
                )

        self._function = fn
        self._function_kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        return self._function(x, **self._function_kwargs)
    
    