# core/exceptions.py

class VivariumCVError(Exception):
    """Base exception for all project errors."""
    pass

class DetectorInitError(VivariumCVError):
    """Raised when model weights fail to load."""
    pass

class InferenceError(VivariumCVError):
    """Raised when model inference fails."""
    pass

class LevelEstimationError(VivariumCVError):
    """Raised when HSV masking or contour extraction fails."""
    pass

class ROIError(VivariumCVError):
    """Raised when a requested ROI zone is not found in config."""
    pass

class CameraError(VivariumCVError):
    """Raised when frame capture from camera fails."""
    pass