#!/usr/bin/env python3
"""
batch_predict.py

Backward compatibility wrapper for scripts/predict_batch.py
Redirects to the new location while maintaining the same interface.

For new usage, prefer:
    python scripts/predict_batch.py

This wrapper is kept for backward compatibility with existing workflows.
"""

import sys
import os

# Add scripts directory to path
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.insert(0, scripts_dir)

# Import and run the actual script
from predict_batch import main

if __name__ == "__main__":
    print("ℹ️  Note: Using backward compatibility wrapper")
    print("   For new usage: python scripts/predict_batch.py\n")
    main()
