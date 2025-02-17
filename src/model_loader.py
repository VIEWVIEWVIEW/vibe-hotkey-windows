from typing import Optional, ClassVar
from PySide6.QtCore import QThread, Signal
from faster_whisper import WhisperModel


class ModelLoaderThread(QThread):
    """
    A thread class for loading Whisper models asynchronously.

    This class handles loading both CUDA and CPU Whisper models in a separate thread
    to prevent blocking the main application. It provides progress updates through signals
    and supports graceful cancellation of the loading process.

    Signals:
        finished (WhisperModel): Emitted when model loading completes successfully
        error (str): Emitted if an error occurs during model loading
        progress (str): Emitted to provide status updates during loading

    Attributes:
        model_name (str): Name/size of the Whisper model to load
        device_mode (str): Either "cuda" or "cpu" to specify device type
        cuda_device (int): CUDA device ID to use when device_mode is "cuda"
    """

    finished: ClassVar[Signal] = Signal(WhisperModel)
    error: ClassVar[Signal] = Signal(str)
    progress: ClassVar[Signal] = Signal(str)

    def __init__(self, model_name: str, device_mode: str = "cuda", cuda_device: int = 0) -> None:
        super().__init__()
        self.model_name: str = model_name
        self.device_mode: str = device_mode
        self.cuda_device: int = cuda_device
        self._is_running: bool = True
        self.models_dir: Optional[str] = None

    def run(self) -> None:
        try:
            if not self._is_running:
                return

            if self.device_mode == "cuda":
                self.progress.emit(f"Loading {self.model_name} model with CUDA (Device {self.cuda_device})...")
                # faster-whisper expects just "cuda" for default device (0) or "cuda:N" for specific devices
                device: str = "cuda" if self.cuda_device == 0 else f"cuda:{self.cuda_device}"
                model: WhisperModel = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type="float16",
                    download_root=self.models_dir,  # Use custom models directory
                    local_files_only=False,
                )

                if self._is_running:
                    self.progress.emit("Model loaded successfully!")
                    self.finished.emit(model)
            else:
                self.progress.emit(f"Loading {self.model_name} model in CPU mode...")
                model: WhisperModel = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=self.models_dir,  # Use custom models directory
                    local_files_only=False,
                )
                if self._is_running:
                    self.progress.emit("Model loaded successfully (CPU)!")
                    self.finished.emit(model)

        except Exception as e:
            if not self._is_running:
                return
            error_msg: str = f"Failed to load model: {str(e)}"
            print(error_msg)  # Log to console
            self.error.emit(error_msg)

    def stop(self) -> None:
        self._is_running = False
        self.wait()  # Wait for the thread to finish
