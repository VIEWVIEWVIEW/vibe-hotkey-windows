from typing import List, ClassVar
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
import time

class LoadingIconThread(QThread):
    update_icon: ClassVar[Signal] = Signal(QIcon)

    def __init__(self) -> None:
        super().__init__()
        self.is_running: bool = True
        self.frames: List[QIcon] = []
        self._create_loading_frames()

    def _create_loading_frames(self) -> None:
        # Create 8 frames for the loading animation
        colors: List[tuple[int, int, int]] = [
            (255, 0, 0),      # Red
            (255, 128, 0),    # Orange
            (255, 255, 0),    # Yellow
            (128, 255, 0),    # Light green
            (0, 255, 0),      # Green
            (0, 255, 255),    # Cyan
            (0, 128, 255),    # Light blue
            (0, 0, 255),      # Blue
        ]
        
        for color in colors:
            pixmap: QPixmap = QPixmap(64, 64)
            pixmap.fill(Qt.transparent)
            painter: QPainter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(*color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(4, 4, 56, 56)
            painter.end()
            self.frames.append(QIcon(pixmap))

    def run(self) -> None:
        frame_index: int = 0
        while self.is_running:
            self.update_icon.emit(self.frames[frame_index])
            frame_index = (frame_index + 1) % len(self.frames)
            time.sleep(0.1)  # 100ms per frame

    def stop(self) -> None:
        self.is_running = False
        self.wait()