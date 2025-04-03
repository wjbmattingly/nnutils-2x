#!/usr/bin/env python
"""Script to run all tests for the nnutils-2x package."""

import sys
import pytest

if __name__ == "__main__":
    # Let pytest handle its own arguments from pytest.ini
    sys.exit(pytest.main()) 