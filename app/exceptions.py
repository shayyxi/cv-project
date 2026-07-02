class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""


class DownloadError(PipelineError):
    """Raised when image download fails."""


class ValidationError(PipelineError):
    """Raised when downloaded or processed data is invalid."""


class StorageError(PipelineError):
    """Raised when file/object storage operation fails."""


class VisionError(PipelineError):
    """Raised when vision processing fails."""


class DeliveryError(PipelineError):
    """Raised when WordPress delivery fails."""