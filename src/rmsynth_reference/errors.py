class RMSynthError(Exception):
    """Base class for expected RMSynth errors."""


class ValidationError(RMSynthError):
    """Input is outside the documented data model."""


class ResourceLimitError(RMSynthError):
    """A documented resource limit was exceeded."""


class VerificationError(RMSynthError):
    """An internally produced circuit failed verification."""
