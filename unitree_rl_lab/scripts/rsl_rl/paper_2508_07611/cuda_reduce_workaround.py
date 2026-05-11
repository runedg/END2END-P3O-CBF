"""CUDA reduction workaround isolated to End2EndP3O entrypoints."""

from __future__ import annotations

import torch


_PATCHED = False
_ORIG_TORCH_ALLCLOSE = torch.allclose
_ORIG_TORCH_ALL = torch.all
_ORIG_TORCH_ANY = torch.any
_ORIG_TENSOR_ALL = torch.Tensor.all
_ORIG_TENSOR_ANY = torch.Tensor.any


def _normalize_reduce_dims(tensor: torch.Tensor, dim):
    if dim is None:
        return tuple(range(tensor.ndim))
    if isinstance(dim, tuple):
        return tuple(d if d >= 0 else tensor.ndim + d for d in dim)
    return (dim if dim >= 0 else tensor.ndim + dim,)


def _safe_cuda_any(input_tensor: torch.Tensor, dim=None, keepdim: bool = False):
    dims = _normalize_reduce_dims(input_tensor, dim)
    predicate = input_tensor if input_tensor.dtype == torch.bool else input_tensor != 0
    reduced = predicate.to(torch.int32).sum(dim=dims, keepdim=keepdim)
    return reduced > 0


def _safe_cuda_all(input_tensor: torch.Tensor, dim=None, keepdim: bool = False):
    dims = _normalize_reduce_dims(input_tensor, dim)
    predicate = input_tensor if input_tensor.dtype == torch.bool else input_tensor != 0
    reduced = predicate.to(torch.int32).sum(dim=dims, keepdim=keepdim)
    expected = 1
    for reduce_dim in dims:
        expected *= input_tensor.shape[reduce_dim]
    return reduced == expected


def _safe_allclose(input_tensor, other_tensor, *args, **kwargs):
    if (
        isinstance(input_tensor, torch.Tensor)
        and isinstance(other_tensor, torch.Tensor)
        and input_tensor.is_cuda
        and other_tensor.is_cuda
    ):
        close = torch.isclose(input_tensor, other_tensor, *args, **kwargs)
        return _safe_cuda_all(close)
    return _ORIG_TORCH_ALLCLOSE(input_tensor, other_tensor, *args, **kwargs)


def _safe_all(input_tensor, *args, **kwargs):
    if isinstance(input_tensor, torch.Tensor) and input_tensor.is_cuda:
        dim = kwargs.pop("dim", None)
        keepdim = kwargs.pop("keepdim", False)
        if kwargs:
            return _ORIG_TORCH_ALL(input_tensor, dim=dim, keepdim=keepdim, **kwargs)
        return _safe_cuda_all(input_tensor, dim=dim, keepdim=keepdim)
    return _ORIG_TORCH_ALL(input_tensor, *args, **kwargs)


def _safe_any(input_tensor, *args, **kwargs):
    if isinstance(input_tensor, torch.Tensor) and input_tensor.is_cuda:
        dim = kwargs.pop("dim", None)
        keepdim = kwargs.pop("keepdim", False)
        if kwargs:
            return _ORIG_TORCH_ANY(input_tensor, dim=dim, keepdim=keepdim, **kwargs)
        return _safe_cuda_any(input_tensor, dim=dim, keepdim=keepdim)
    return _ORIG_TORCH_ANY(input_tensor, *args, **kwargs)


def _safe_tensor_all(self, *args, **kwargs):
    if isinstance(self, torch.Tensor) and self.is_cuda:
        dim = kwargs.pop("dim", None)
        keepdim = kwargs.pop("keepdim", False)
        if kwargs:
            return _ORIG_TENSOR_ALL(self, dim=dim, keepdim=keepdim, **kwargs)
        return _safe_cuda_all(self, dim=dim, keepdim=keepdim)
    return _ORIG_TENSOR_ALL(self, *args, **kwargs)


def _safe_tensor_any(self, *args, **kwargs):
    if isinstance(self, torch.Tensor) and self.is_cuda:
        dim = kwargs.pop("dim", None)
        keepdim = kwargs.pop("keepdim", False)
        if kwargs:
            return _ORIG_TENSOR_ANY(self, dim=dim, keepdim=keepdim, **kwargs)
        return _safe_cuda_any(self, dim=dim, keepdim=keepdim)
    return _ORIG_TENSOR_ANY(self, *args, **kwargs)


def apply_cuda_reduce_workaround() -> None:
    global _PATCHED
    if _PATCHED:
        return
    torch.allclose = _safe_allclose
    torch.all = _safe_all
    torch.any = _safe_any
    torch.Tensor.all = _safe_tensor_all
    torch.Tensor.any = _safe_tensor_any
    _PATCHED = True
