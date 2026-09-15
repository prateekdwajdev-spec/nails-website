class BaseNailException(Exception):
    """Base exception for nail microservice domain errors."""
    pass

class CalibrationError(BaseNailException):
    """Raised when A4 reference paper calibration fails."""
    pass

class HandLandmarkError(BaseNailException):
    """Raised when hand landmark detection fails."""
    pass

class SegmentationError(BaseNailException):
    """Raised when MobileSAM fingernail segmentation fails."""
    pass

class MeasurementError(BaseNailException):
    """Raised when dimension calculation fails."""
    pass