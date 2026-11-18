"""Diagnostics and exceptions for symbolic Airy solves."""

from dataclasses import dataclass, field


class AirySolverError(Exception):
    """Base exception for Airy solver errors."""


class InconsistentBoundaryError(AirySolverError):
    """Raised when boundary equations have no solution."""


class UnderdeterminedSystemError(AirySolverError):
    """Raised when boundary equations do not determine all coefficients."""


@dataclass(frozen=True)
class SolveDiagnostics:
    status: str
    n_unknowns: int
    n_boundary_equations: int
    matrix_rank: int
    augmented_rank: int
    free_parameters: tuple = field(default_factory=tuple)
    max_symbolic_residual: object = 0
    highest_degree: int | None = None
    n_basis_functions: int = 0
    n_integral_constraints: int = 0


@dataclass(frozen=True)
class AirySolution:
    U: object
    sigma_x: object
    sigma_y: object
    tau_xy: object
    coefficients: dict
    diagnostics: SolveDiagnostics
