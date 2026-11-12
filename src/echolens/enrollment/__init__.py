"""
Echolens — Target Speaker Enrollment Module
===========================================
"""

from .builder import (
    EnrollmentBuilder,
    EnrollmentResult,
    EnrollmentError,
    build_enrollment,
)

__all__ = [
    "EnrollmentBuilder",
    "EnrollmentResult",
    "EnrollmentError",
    "build_enrollment",
]
