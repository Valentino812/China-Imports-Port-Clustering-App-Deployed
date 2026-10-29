"""Shared configuration constants for the deployed application.

This deployment uses the four bundled datasets in ``dataset_input`` and a fixed
cluster count, so there is no user-facing K configuration:

* ``FIXED_K`` is the cluster count used by the whole application (K = 5).
* ``DEFAULT_K`` mirrors ``FIXED_K`` for the modelling helpers that accept a
  default value.
* ``OPTIMAL_K`` keeps the evaluated optimum used in the dashboard narrative.
"""

from __future__ import annotations

K_MIN = 2
K_MAX = 8
K_RANGE = range(K_MIN, K_MAX + 1)

FIXED_K = 5
DEFAULT_K = FIXED_K
OPTIMAL_K = 5
