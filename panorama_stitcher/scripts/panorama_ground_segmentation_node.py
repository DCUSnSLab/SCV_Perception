#!/usr/bin/env python3
"""Executable wrapper for the Python ground segmentation node."""

# flake8: noqa: E402 -- native thread limits must precede NumPy import.

import os


_CPU_THREADS = max(1, int(os.environ.get('PANORAMA_CPU_THREADS', '2')))
for _variable in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = str(_CPU_THREADS)

from panorama_stitcher_py.ground_segmentation_node import main


if __name__ == '__main__':
    main()
