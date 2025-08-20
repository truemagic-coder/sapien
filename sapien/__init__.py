# src/sapien/__init__.py
"""
Sapien – a lightweight temporal knowledge‑graph client

The package exposes the public API that you’ll use in your code:

>>> from sapien import SapienClient, SapienConfig, CollectionNames
"""

from .client import SapienClient, SapienConfig, CollectionNames

__all__ = ["SapienClient", "SapienConfig", "CollectionNames"]
