"""Data adapters built on top of prepared_data artifacts."""

from .loaders import get_client_ids, make_client_loaders, make_open_loader, make_test_loader

__all__ = [
    "get_client_ids",
    "make_client_loaders",
    "make_open_loader",
    "make_test_loader",
]
