#!/usr/bin/env python3
"""VespAI test runner based on standard unittest discovery."""

import os
import sys
import unittest

def run_tests():
    """Discover and run the repository test suite."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(project_root, 'tests')

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if not os.path.isdir(tests_dir):
        print("Tests directory not found!")
        return 1

    suite = unittest.defaultTestLoader.discover(
        start_dir=tests_dir,
        pattern='test_*.py',
        top_level_dir=project_root,
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)