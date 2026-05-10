"""Domain-specific exceptions."""


class ForecastingSystemError(Exception):
    """Base exception for the project."""


class SchemaValidationError(ForecastingSystemError):
    """Raised when structured data does not satisfy a schema contract."""


class PlaceholderNotImplementedError(ForecastingSystemError):
    """Raised by scaffolded functions that are intentionally unimplemented."""
