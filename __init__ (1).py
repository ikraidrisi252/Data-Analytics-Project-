"""
Database module for Retail Pulse Analytics.
Provides SQL schema definitions and mock data generation.
"""

from .mock_data import generate_mock_dataset, init_database

__all__ = ["generate_mock_dataset", "init_database"]
