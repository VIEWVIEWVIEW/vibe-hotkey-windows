import json
import sys
import signal
from pathlib import Path
import winsound
from pynput import keyboard
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QSystemTrayIcon,
    QMenu,
    QVBoxLayout,
    QLabel,
    QPushButton,
)
from PySide6.QtGui import QIcon, QPixmap, QAction, QActionGroup, QPainter, QColor
from PySide6.QtCore import Qt, QTimer
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as write_wav
import tempfile
import pyperclip
import os
import time
from cuda_utils import set_cuda_paths, check_cuda_availability
from model_loader import ModelLoaderThread
from loader_icon_thread import LoadingIconThread


set_cuda_paths()


class SetNewRecordingShortcut(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Hotkey")
        self.setFixedSize(300, 150)
        # Set dialog flags to make it behave as a tool window
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)
        layout = QVBoxLayout(self)
        # Current hotkey display
        self.current_label = QLabel("Current Hotkey:")
        layout.addWidget(self.current_label, alignment=Qt.AlignCenter)
        self.current_hotkey = QLabel()
        layout.addWidget(self.current_hotkey, alignment=Qt.AlignCenter)
        # New hotkey input
        self.input_label = QLabel("Press the keys for your new hotkey:")
        layout.addWidget(self.input_label, alignment=Qt.AlignCenter)
        self.hotkey_display = QLabel("Press 'Start Recording'")
        layout.addWidget(self.hotkey_display, alignment=Qt.AlignCenter)
        self.record_button = QPushButton("Start Recording")
        layout.addWidget(self.record_button, alignment=Qt.AlignCenter)
        # Add OK button
        self.ok_button = QPushButton("OK")
        layout.addWidget(self.ok_button, alignment=Qt.AlignCenter)
        self.ok_button.clicked.connect(self.hide)

    def closeEvent(self, event):
        # Just hide the dialog instead of closing the application
        self.hide()
        event.ignore()


class HotkeyApp:
    def __init__(self):
        self.app = QApplication(sys.argv)

        # create gray square icon
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(128, 128, 128))  # Medium gray
        self.gray_icon = QIcon(pixmap)

        # create red circle icon
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(255, 0, 0))  # Red
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        painter.end()

        self.red_circle_icon = QIcon(pixmap)

        # create blue circle icon for transcription
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(0, 0, 255))  # Solid blue
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        painter.end()
        self.blue_circle_icon = QIcon(pixmap)

        self.config_file = Path("config.json")
        self.config = self.load_config()
        # Set models directory before anything else
        self.models_dir = get_models_directory()
        # Check CUDA availability
        self.cuda_device_count = check_cuda_availability()
        # Always start with CPU mode if no CUDA devices are available
        if self.cuda_device_count == 0:
            print("CUDA is not available. Using CPU mode.")
            self.device_mode = "cpu"
            self.cuda_device = 0
        else:
            print(f"Found {self.cuda_device_count} CUDA device(s)")
            # Only use CUDA from config if it was previously set and CUDA is available
            config_device_mode = self.config.get("device_mode", "cuda")
            self.device_mode = config_device_mode if config_device_mode == "cuda" else "cpu"
            self.cuda_device = self.config.get("cuda_device", 0)
            # Validate cuda_device against available devices
            if self.cuda_device >= self.cuda_device_count:
                print(f"Configured CUDA device {self.cuda_device} not available, using device 0")
                self.cuda_device = 0
        # Initialize loading animation
        self.loading_thread = LoadingIconThread()
        self.loading_thread.update_icon.connect(self._update_tray_icon)
        self.loading_thread.start()

        # create pointer which will later point to the transcription thread
        self.transcribing_thread = None

        # Initialize hotkey state
        self.hotkey = self.config["hotkey"]
        self.pressed_keys = set()  # Track currently pressed keys
        self.recording_hotkey = False  # Flag for hotkey recording mode
        self.temp_hotkey = set()  # Temporary storage for new hotkey
        self._toggle_recording_func = None  # Store reference to toggle_recording function
        # Initialize sound threads
        sounds_dir = Path("sounds")
        self.sound_files = {
            "start_record": str(sounds_dir / "start_record.wav"),
            "stop_record": str(sounds_dir / "stop_record.wav"),
            "transcription_done": str(sounds_dir / "transcription_done.wav"),
            "transcription_empty": str(sounds_dir / "transcription_empty.wav"),
        }
        self.current_model = self.config["model"]
        self.current_language = self.config["language"]
        self.available_languages = self.config["available_languages"]
        self.initial_prompt = self.config["initial_prompt"]
        self.available_models = self.config["available_models"]
        self.listener = None
        self.tray = None
        # Audio recording state
        self.is_recording = False
        self.recording_data = []
        self.sample_rate = 16000
        self.last_trigger_time = 0
        self.trigger_cooldown = 0.3
        # Create the dialog but don't show it yet
        self.dialog = SetNewRecordingShortcut()
        self.model = None
        self.model_loader = None
        self.auto_paste = self.config["auto_paste"]
        self.setup_listener()
        self.create_tray_icon()
        # Start loading animation
        self.loading_thread.start()
        # Setup signal handling. This is used to handle the Ctrl+C signal.
        signal.signal(signal.SIGINT, lambda x, y: self.handle_sigint())
        # Create timer to check for signals
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.check_signal)
        self.check_timer.start(500)  # Check every 500ms
        # Load model after everything else is setup
        self.load_whisper_model()
        self.transcribing = False

    def handle_sigint(self):
        print("Caught Ctrl+C, closing application...")
        self.quit_application()
        sys.exit(0)

    def check_signal(self):
        # This method is called periodically to allow signal processing
        self.check_timer.start(500)  # Restart timer

    def load_config(self):
        default_config = {
            "sound_settings": {
                "start_record": True,
                "stop_record": True,
                "transcription_done": True,
                "transcription_empty": True,
            },
            "auto_paste": True,
            "available_languages": [
                {"code": "de", "name": "German"},
                {"code": "en", "name": "English"},
                {"code": "fr", "name": "French"},
            ],
            "language": "en",
            "model": "tiny",
            "device_mode": "cuda",
            "cuda_device": 0,
            "available_models": [
                "tiny",
                "tiny.en",
                "base",
                "base.en",
                "small",
                "small.en",
                "distil-small.en",
                "medium",
                "medium.en",
                "distil-medium.en",
                "large-v1",
                "large-v2",
                "large-v3",
                "large",
                "distil-large-v2",
                "distil-large-v3",
                "large-v3-turbo",
                "turbo",
            ],
            "hotkey": {"ctrl", "shift", "space"},
            "initial_prompt": "The following is a transcription of spoken {language}:",
        }
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                try:
                    config = json.load(f)
                    # Convert hotkey list back to set if it exists
                    if "hotkey" in config:
                        config["hotkey"] = set(config["hotkey"])
                    # Merge with defaults to ensure all required keys exist
                    return {**default_config, **config}
                except json.JSONDecodeError:
                    return default_config
        return default_config

    def save_config(self):
        config = {
            "hotkey": list(self.hotkey),  # Convert set to list for JSON
            "model": self.current_model,
            "language": self.current_language,
            "device_mode": self.device_mode,
            "cuda_device": self.cuda_device if self.device_mode == "cuda" else 0,
            "available_languages": self.available_languages,
            "initial_prompt": self.initial_prompt.format(language=self.current_language),
            "auto_paste": self.auto_paste,  # Add auto-paste setting
            "sound_settings": self.config.get(
                "sound_settings",
                {
                    "start_record": True,
                    "stop_record": True,
                    "transcription_done": True,
                    "transcription_empty": True,
                },
            ),
        }
        # debug print the initial prompt
        with open(self.config_file, "w") as f:
            json.dump(config, f)

    def save_hotkey(self):
        # Update to use save_config instead
        self.save_config()

    def on_press(self, key):
        try:
            # Get standardized key representation
            key_str = self._get_key_string(key)
            if not key_str:
                return
            if self.recording_hotkey and self.dialog:
                # If all keys were released and we're pressing a new key (excluding mouse buttons),
                # clear the temp hotkey
                if len(self.pressed_keys) == 0 and key_str not in ["mouse1", "mouse2"]:
                    self.temp_hotkey.clear()
                # Handle hotkey recording mode
                self.temp_hotkey.add(key_str)
                self.pressed_keys.add(key_str)  # Also track in pressed_keys during recording
                self.dialog.hotkey_display.setText(" + ".join(sorted(self.temp_hotkey)))
            else:
                # Normal operation mode
                self.pressed_keys.add(key_str)
                self._check_hotkey()
        except Exception as e:
            print(f"Error in on_press: {str(e)}")
            import traceback

            traceback.print_exc()  # Print full stack trace for debugging

    def on_release(self, key):
        try:
            key_str = self._get_key_string(key)
            if not key_str:
                return
            # Remove from pressed keys
            self.pressed_keys.discard(key_str)
            # During recording, we don't remove from temp_hotkey on release
            # This allows building multi-key combinations
        except Exception as e:
            print(f"Error in on_release: {e}")

    def _get_key_string(self, key):
        """Convert a key to a standardized string representation."""
        # Handle special keys first
        if hasattr(key, "name"):
            key_str = str(key).lower()
            # Map special keys more precisely
            key_map = {
                "key.shift": "shift",
                "key.shift_l": "shift",
                "key.shift_r": "shift",
                "key.ctrl": "ctrl",
                "key.ctrl_l": "ctrl",
                "key.ctrl_r": "ctrl",
                "key.alt": "alt",
                "key.alt_l": "alt",
                "key.alt_r": "alt",
            }
            mapped_key = key_map.get(key_str)
            if mapped_key:
                return mapped_key
            # For other special keys, remove 'key.' prefix
            return key_str.replace("key.", "")
        # Handle letters and numbers
        if hasattr(key, "vk") and ((65 <= key.vk <= 90) or (48 <= key.vk <= 57)):  # A-Z or 0-9
            return chr(key.vk).lower()  # Convert to lowercase for consistency
        # Handle normal characters as fallback
        if hasattr(key, "char") and key.char:
            if key.char.isprintable():  # Only handle printable characters
                return key.char.lower()  # Convert to lowercase for consistency
        return None

    def _check_hotkey(self):
        """Check if the current pressed keys match the hotkey exactly."""
        if not self.hotkey:
            return
        # Only trigger if:
        # 1. The number of pressed keys exactly matches the hotkey
        # 2. All required hotkey keys are present in pressed_keys
        # 3. At least one non-modifier key is in the hotkey (to prevent triggering on just modifiers)
        modifier_keys = {"ctrl", "shift", "alt"}
        has_non_modifier = any(key not in modifier_keys for key in self.hotkey)
        if len(self.pressed_keys) == len(self.hotkey) and self.pressed_keys == self.hotkey and has_non_modifier:
            self.trigger_action()

    def play_sound(self, sound_name):
        """Play a sound asynchronously."""
        # Check if sound is enabled in settings
        if not self.config["sound_settings"].get(sound_name, True):
            return
        # Play sound asynchronously
        if sound_name in self.sound_files:
            sound_file = str(Path(self.sound_files[sound_name]).resolve())
            if Path(sound_file).exists():
                winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                print(f"Error: Sound file not found: {sound_file}")
        else:
            print(f"Warning: Sound '{sound_name}' not found in sound_files")

    def setup_listener(self):
        if self.listener:
            self.listener.stop()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def start_recording(self):
        self.recording_data = []
        self.is_recording = True
        # Show red circle while recording
        self.tray.setIcon(self.red_circle_icon)
        # Play start sound
        self.play_sound("start_record")

        def callback(indata, frames, time, status):
            if self.is_recording:
                self.recording_data.append(indata.copy())

        # Start recording stream
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback)
        self.stream.start()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        # Clear any pressed keys
        self.pressed_keys.clear()
        # Play stop recording sound
        self.play_sound("stop_record")
        # Check if model is loaded
        if not self.model:
            print("Cannot transcribe - model not loaded yet")
            # Set default gray square icon since we're not transcribing
            self.tray.setIcon(self.gray_icon)
            return
        # Set blue circle for transcription
        self.transcribing = True
        self.tray.setIcon(self.blue_circle_icon)

        self.update_tray_menu()
        # Process the recording
        if self.recording_data:
            try:
                # Combine all chunks
                audio_data = np.concatenate(self.recording_data, axis=0)
                # Calculate audio length in seconds
                audio_length = len(audio_data) / self.sample_rate
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    write_wav(temp_file.name, self.sample_rate, audio_data)
                    # Transcribe
                    print(f"Transcribing {audio_length:.1f} seconds of audio...")
                    segments, info = self.model.transcribe(
                        temp_file.name,
                        beam_size=5,
                        language=self.current_language,
                        initial_prompt=self.initial_prompt.format(language=self.current_language),
                    )
                    # Combine all segments
                    transcription = " ".join([segment.text for segment in segments])
                    # Copy to clipboard
                    pyperclip.copy(transcription)
                    # Auto-paste if enabled
                    if self.auto_paste:
                        keyboard.Controller().press(keyboard.Key.ctrl)
                        keyboard.Controller().press("v")
                        keyboard.Controller().release("v")
                        keyboard.Controller().release(keyboard.Key.ctrl)
                    # Print to console
                    print("Transcription:")
                    print(transcription)
                    print(f"Language: {info.language} (confidence: {info.language_probability:.2%})")
                    print("(Copied to clipboard)")
                    # Play appropriate sound based on transcription content
                    if transcription.strip():
                        self.play_sound("transcription_done")
                    else:
                        self.play_sound("transcription_empty")
            except Exception as e:
                print(f"Error during transcription: {e}")
            finally:
                # Always restore default icon
                self.transcribing = False
                self.tray.setIcon(self.gray_icon)
                self.update_tray_menu()

    def trigger_action(self):
        current_time = time.time()
        # Don't allow recording if hotkey dialog is open
        if self.dialog.isVisible():
            print("Cannot start recording while hotkey dialog is open")
            return
        # Strict hotkey checking:
        # 1. Must have exactly the same number of keys
        # 2. Must contain ALL keys from the hotkey
        # 3. Must not contain any extra keys
        if len(self.pressed_keys) != len(self.hotkey) or not all(key in self.pressed_keys for key in self.hotkey) or not all(key in self.hotkey for key in self.pressed_keys):
            return
        # Only apply cooldown when we're already recording to prevent accidental double-triggers
        if self.is_recording and current_time - self.last_trigger_time < self.trigger_cooldown:
            return
        self.last_trigger_time = current_time
        self._last_trigger_keys = self.pressed_keys.copy()
        if not self.is_recording:
            # Only start if we're not already recording or transcribing
            if self.transcribing:
                print("[Hotkey Pressed] Cannot start recording while transcribing...")
                return
            print("[Hotkey Pressed] Starting recording...")
            self.start_recording()
        else:
            print("[Hotkey Pressed] Stopping recording...")
            self.stop_recording()

    def change_hotkey(self):
        # Don't allow hotkey changes during recording or transcription
        if self.is_recording:
            print("Cannot change hotkey while recording is in progress")
            return
        if self.transcribing:
            print("Cannot change hotkey while transcription is in progress")
            return
        if not self.dialog.isVisible():
            self.dialog.current_hotkey.setText(" + ".join(sorted(self.hotkey)))
            self.dialog.hotkey_display.setText("Press 'Start Recording'")
            # Disconnect previous handler if it exists
            if self._toggle_recording_func is not None:
                try:
                    self.dialog.record_button.clicked.disconnect(self._toggle_recording_func)
                except Exception:
                    pass

            def toggle_recording():
                # Don't allow hotkey recording if audio recording or transcription is in progress
                if self.is_recording or self.transcribing:
                    print("Cannot change hotkey while recording or transcription is in progress")
                    return
                self.recording_hotkey = not self.recording_hotkey
                if self.recording_hotkey:
                    self.dialog.record_button.setText("Stop Recording")
                    self.temp_hotkey.clear()
                    self.dialog.hotkey_display.setText("Recording...")
                    # Unfocus the button to prevent space/enter from stopping the recording
                    self.dialog.record_button.clearFocus()
                else:
                    self.dialog.record_button.setText("Start Recording")
                    if self.temp_hotkey:
                        self.hotkey = self.temp_hotkey.copy()
                        self.save_config()
                        self.update_tray_menu()
                        self.dialog.current_hotkey.setText(" + ".join(sorted(self.hotkey)))
                        self.dialog.hotkey_display.setText("Press 'Start Recording'")

            # Store reference to new handler
            self._toggle_recording_func = toggle_recording
            self.dialog.record_button.clicked.connect(self._toggle_recording_func)
            self.dialog.show()

    def toggle_shortcut(self, shortcut_path, checked):
        """Create or remove a shortcut at the specified path."""
        try:
            if checked:
                # Create shortcut using win32com
                from win32com.client import Dispatch
                import pythoncom

                # Initialize COM in this thread
                pythoncom.CoInitialize()

                python_path = str(Path(sys.executable).resolve())
                script_path = str(Path(__file__).resolve())
                working_dir = str(Path(script_path).parent)

                # Create Windows shell shortcut
                shell = Dispatch("WScript.Shell")
                shortcut = shell.CreateShortCut(str(shortcut_path))

                # Configure shortcut properties
                shortcut.TargetPath = python_path
                shortcut.Arguments = f'"{script_path}"'
                shortcut.WorkingDirectory = working_dir
                shortcut.IconLocation = python_path + ",0"  # Use Python icon
                shortcut.WindowStyle = 7  # Minimized window
                shortcut.Description = "VibeHotkeyWindows Voice Recording Tool"

                # Create parent directory if it doesn't exist
                shortcut_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the shortcut
                shortcut.save()

                print(f"Added shortcut at: {shortcut_path}")

                # Cleanup COM
                pythoncom.CoUninitialize()

            else:
                if shortcut_path.exists():
                    shortcut_path.unlink()
                    print(f"Removed shortcut: {shortcut_path}")
                    # Remove parent directory if empty
                    if shortcut_path.parent.exists() and not any(shortcut_path.parent.iterdir()):
                        shortcut_path.parent.rmdir()

        except Exception as e:
            print(f"Failed to manage shortcut at {shortcut_path}: {e}")
            import traceback

            traceback.print_exc()

    def is_shortcut_enabled(self, shortcut_path):
        """Check if a shortcut exists at the specified path."""
        return shortcut_path.exists()

    def toggle_autorun(self, checked):
        """Toggle startup shortcut."""
        startup_folder = Path(os.path.expandvars("%APPDATA%/Microsoft/Windows/Start Menu/Programs/Startup"))
        shortcut_path = startup_folder / "VibeHotkeyWindows.lnk"
        self.toggle_shortcut(shortcut_path, checked)

    def toggle_start_menu(self, checked):
        """Toggle start menu shortcut."""
        start_menu = Path(os.path.expandvars("%APPDATA%/Microsoft/Windows/Start Menu/Programs"))
        shortcut_path = start_menu / "VibeHotkeyWindows" / "VibeHotkeyWindows.lnk"
        self.toggle_shortcut(shortcut_path, checked)

    def load_whisper_model(self):
        if self.model_loader and self.model_loader.isRunning():
            return
        self.model = None  # Clear current model while loading
        self.model_loader = ModelLoaderThread(
            self.current_model,
            self.device_mode,
            cuda_device=self.cuda_device if self.device_mode == "cuda" else 0,
        )
        # Pass models directory to ModelLoaderThread
        self.model_loader.models_dir = self.models_dir
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.error.connect(self.on_model_error)
        self.model_loader.progress.connect(self.on_model_progress)
        self.model_loader.start()
        # Update menu to show loading status
        self.update_tray_menu("Loading...")

    def on_model_loaded(self, model):
        """Handle successful model loading."""
        try:
            self.model = model
            # Always stop loading animation and clear reference
            if self.loading_thread:
                self.loading_thread.stop()
                self.loading_thread = None
            # Set final icon
            self.tray.setIcon(self.gray_icon)
            # Refresh the menu to update sizes after model download
            if hasattr(self, "model_menu"):
                old_menu = self.model_menu
                new_menu = self.create_model_submenu(old_menu.parent())
                old_menu.parent().insertMenu(old_menu.menuAction(), new_menu)
                old_menu.parent().removeAction(old_menu.menuAction())
            self.update_tray_menu()
        except Exception as e:
            # Log any errors during cleanup
            with open("error.log", "a") as f:
                f.write(f"Error in on_model_loaded: {str(e)}\n")

    def on_model_error(self, error):
        """Handle model loading errors."""
        try:
            self.model = None
            # Always stop loading animation and clear reference
            if self.loading_thread:
                self.loading_thread.stop()
                self.loading_thread = None
            # Set final icon
            self.tray.setIcon(self.gray_icon)
            self.update_tray_menu(f"Error: {error}")
            # Log the error
            with open("error.log", "a") as f:
                f.write(f"Model loading error: {error}\n")
        except Exception as e:
            # Log any errors during cleanup
            with open("error.log", "a") as f:
                f.write(f"Error in on_model_error: {str(e)}\n")

    def on_model_progress(self, message):
        print(f"{message}")
        self.update_tray_menu(message)

    def change_model(self, model_name):
        # Don't allow model changes while loading
        if self.model_loader and self.model_loader.isRunning():
            print("Cannot change model while another model is being loaded")
            return
        if model_name != self.current_model:
            self.current_model = model_name
            self.save_config()  # Save when model changes
            self.model = None  # Clear current model
            # Stop any existing loading animation
            if self.loading_thread and self.loading_thread.isRunning():
                self.loading_thread.stop()
                self.loading_thread = None
            # Create and start a new loading animation thread
            self.loading_thread = LoadingIconThread()
            self.loading_thread.update_icon.connect(self._update_tray_icon)
            self.loading_thread.start()
            # Recreate the model submenu to refresh sizes
            if hasattr(self, "model_menu"):
                old_menu = self.model_menu
                new_menu = self.create_model_submenu(old_menu.parent())
                old_menu.parent().insertMenu(old_menu.menuAction(), new_menu)
                old_menu.parent().removeAction(old_menu.menuAction())
            self.load_whisper_model()

    def create_model_submenu(self, parent_menu):
        class ModelMenu(QMenu):
            def __init__(self, title, parent, app):
                super().__init__(title, parent)
                self.app = app
                # Add info label at the top
                info_action = QAction("Right click to delete model", self)
                info_action.setEnabled(False)  # Make it non-clickable
                self.addAction(info_action)
                self.addSeparator()  # Add separator after the info label

            # hijack the mouse event so we can use right click to delete teh model
            def mouseReleaseEvent(self, event):
                if event.button() == Qt.RightButton:
                    # Use position() instead of pos() for Qt6 compatibility
                    action = self.actionAt(event.position().toPoint())
                    if action and hasattr(action, "model_name"):
                        # Don't allow deletion of current model
                        if action.model_name == self.app.current_model:
                            return
                        # Get the models directory from app settings
                        models_dir = Path(os.getenv("LOCALAPPDATA")) / "VibeHotkeyWindows" / "models"
                        dir_name = f"models--Systran--faster-whisper-{action.model_name}"
                        model_path = models_dir / dir_name
                        if model_path.exists():
                            print(f"Deleting model {action.model_name} from {model_path}")
                            try:
                                import shutil

                                shutil.rmtree(model_path)
                                print(f"Successfully deleted model {action.model_name}")
                                # Refresh the menu to update sizes
                                old_menu = self.app.model_menu
                                new_menu = self.app.create_model_submenu(old_menu.parent())
                                old_menu.parent().insertMenu(old_menu.menuAction(), new_menu)
                                old_menu.parent().removeAction(old_menu.menuAction())
                            except Exception as e:
                                print(f"Error deleting model: {e}")
                        else:
                            print(f"Model {action.model_name} is not downloaded")
                        self.close()
                        return
                super().mouseReleaseEvent(event)

        model_menu = ModelMenu("Select Model", parent_menu, self)
        model_group = QActionGroup(model_menu)
        model_group.setExclusive(True)

        def get_model_dir_size(model_name):
            """Get the size of the model directory if it exists."""
            dir_name = f"models--Systran--faster-whisper-{model_name}"
            model_path = Path(self.models_dir) / dir_name
            if not model_path.exists():
                return None
            total_size = 0
            for path in model_path.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
            for unit in ["B", "KB", "MB", "GB"]:
                if total_size < 1024:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024
            return f"{total_size:.1f}TB"

        class ModelAction(QAction):
            def __init__(self, text, parent, model_name):
                super().__init__(text, parent)
                self.model_name = model_name

        for model in self.available_models:
            size = get_model_dir_size(model)
            display_name = f"{model} ({size})" if size else model
            action = ModelAction(display_name, model_menu, model)
            action.setCheckable(True)
            action.setChecked(model == self.current_model)
            # Disable the action if it's the current model
            action.setEnabled(model != self.current_model)
            action.triggered.connect(lambda checked, m=model: self.change_model(m))
            model_group.addAction(action)
            model_menu.addAction(action)
        self.model_menu = model_menu
        return model_menu

    def create_language_submenu(self, parent_menu):
        language_menu = QMenu("Select Language", parent_menu)
        # Create action group for radio buttons
        language_group = QActionGroup(language_menu)
        language_group.setExclusive(True)
        for lang_obj in self.available_languages:
            lang_code = lang_obj["code"]
            lang_name = lang_obj["name"]
            action = QAction(lang_name, language_menu)
            action.setCheckable(True)
            action.setChecked(lang_code == self.current_language)
            action.triggered.connect(lambda checked, lang=lang_code: self.change_language(lang))
            language_group.addAction(action)
            language_menu.addAction(action)
        return language_menu

    def create_sound_settings_submenu(self, parent_menu):
        sound_menu = QMenu("Sound Settings", parent_menu)
        # Create actions for each sound setting
        start_sound = QAction("Start Record Sound", sound_menu)
        start_sound.setCheckable(True)
        start_sound.setChecked(self.config["sound_settings"]["start_record"])
        start_sound.triggered.connect(lambda checked: self.toggle_sound_setting("start_record", checked))
        stop_sound = QAction("Stop Record Sound", sound_menu)
        stop_sound.setCheckable(True)
        stop_sound.setChecked(self.config["sound_settings"]["stop_record"])
        stop_sound.triggered.connect(lambda checked: self.toggle_sound_setting("stop_record", checked))
        done_sound = QAction("Transcription Done Sound", sound_menu)
        done_sound.setCheckable(True)
        done_sound.setChecked(self.config["sound_settings"]["transcription_done"])
        done_sound.triggered.connect(lambda checked: self.toggle_sound_setting("transcription_done", checked))
        empty_sound = QAction("Empty Transcription Sound", sound_menu)
        empty_sound.setCheckable(True)
        empty_sound.setChecked(self.config["sound_settings"]["transcription_empty"])
        empty_sound.triggered.connect(lambda checked: self.toggle_sound_setting("transcription_empty", checked))
        sound_menu.addAction(start_sound)
        sound_menu.addAction(stop_sound)
        sound_menu.addAction(done_sound)
        sound_menu.addAction(empty_sound)
        return sound_menu

    def toggle_sound_setting(self, sound_type, enabled):
        self.config["sound_settings"][sound_type] = enabled
        self.save_config()

    def _update_tray_icon(self, icon):
        if self.tray:
            self.tray.setIcon(icon)

    def create_tray_icon(self):
        # Create system tray icon
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(self.gray_icon)
        self.tray.setVisible(True)
        # Create tray menu
        menu = QMenu()
        # Add model info and submenu
        self.model_action = QAction(f"Model: {self.current_model}", menu)
        self.model_action.setEnabled(False)  # Make it act as a label
        menu.addAction(self.model_action)
        menu.addMenu(self.create_model_submenu(menu))
        menu.addSeparator()
        # Add device mode info and submenu
        self.device_action = QAction(f"Device: {self.device_mode.upper()}", menu)
        self.device_action.setEnabled(False)  # Make it act as a label
        menu.addAction(self.device_action)
        menu.addMenu(self.create_device_submenu(menu))
        menu.addSeparator()
        # Add language info and submenu
        self.language_action = QAction(f"Language: {self.current_language}", menu)
        self.language_action.setEnabled(False)  # Make it act as a label
        menu.addAction(self.language_action)
        menu.addMenu(self.create_language_submenu(menu))
        menu.addSeparator()
        # Add sound settings submenu
        menu.addMenu(self.create_sound_settings_submenu(menu))
        menu.addSeparator()
        # Add auto-paste checkbox
        auto_paste_action = QAction("Auto-paste after transcription", menu)
        auto_paste_action.setCheckable(True)
        auto_paste_action.setChecked(self.auto_paste)
        auto_paste_action.triggered.connect(self.toggle_auto_paste)
        menu.addAction(auto_paste_action)
        # Add autorun checkbox
        autorun_action = QAction("Start with Windows", menu)
        autorun_action.setCheckable(True)
        autorun_action.setChecked(self.is_shortcut_enabled(Path(os.path.expandvars("%APPDATA%/Microsoft/Windows/Start Menu/Programs/Startup")) / "VibeHotkeyWindows.lnk"))
        autorun_action.triggered.connect(self.toggle_autorun)
        menu.addAction(autorun_action)

        # Add start menu shortcut checkbox
        start_menu_action = QAction("Add to Start Menu", menu)
        start_menu_action.setCheckable(True)
        start_menu_action.setChecked(
            self.is_shortcut_enabled(Path(os.path.expandvars("%APPDATA%/Microsoft/Windows/Start Menu/Programs")) / "VibeHotkeyWindows" / "VibeHotkeyWindows.lnk")
        )
        start_menu_action.triggered.connect(self.toggle_start_menu)
        menu.addAction(start_menu_action)

        menu.addSeparator()
        # Add hotkey menu item
        self.hotkey_action = QAction(f"Current Hotkey: {' + '.join(self.hotkey)}", menu)
        self.hotkey_action.triggered.connect(self.change_hotkey)
        menu.addAction(self.hotkey_action)
        # Add exit action
        exit_action = QAction("Exit", menu)
        exit_action.triggered.connect(self.quit_application)
        menu.addAction(exit_action)
        self.tray.setContextMenu(menu)

    def get_language_name(self, lang_code):
        for lang_obj in self.available_languages:
            if lang_obj["code"] == lang_code:
                return lang_obj["name"]
        return lang_code  # fallback to code if name not found

    def update_tray_menu(self, status_message=None):
        _ = self.tray.contextMenu()
        # Update model label with loading/transcribing status
        if not self.model:
            status = f" ({status_message})" if status_message else " (Loading...)"
            # Disable model menu while loading
            if hasattr(self, "model_menu"):
                self.model_menu.setEnabled(False)
            if hasattr(self, "device_menu"):
                self.device_menu.setEnabled(False)
        elif self.transcribing:
            status = " (Transcribing...)"
            # Enable menus after loading
            if hasattr(self, "model_menu"):
                self.model_menu.setEnabled(True)
            if hasattr(self, "device_menu"):
                self.device_menu.setEnabled(True)
        else:
            status = ""
            # Enable menus in normal state
            if hasattr(self, "model_menu"):
                self.model_menu.setEnabled(True)
            if hasattr(self, "device_menu"):
                self.device_menu.setEnabled(True)
        # Update model label
        self.model_action.setText(f"Model: {self.current_model}{status}")
        # Update device label with CUDA device number if applicable
        device_text = f"Device: {self.device_mode.upper()}"
        if self.device_mode == "cuda":
            device_text += f" (Device {self.cuda_device})"
        self.device_action.setText(device_text)
        # Update language label with full name
        current_lang_name = self.get_language_name(self.current_language)
        self.language_action.setText(f"Language: {current_lang_name}")
        # Update hotkey label
        self.hotkey_action.setText(f"Current Hotkey: {' + '.join(self.hotkey)}")

    def change_language(self, language):
        if language != self.current_language:
            self.current_language = language
            self.save_config()
            self.update_tray_menu()

    def quit_application(self):
        if self.is_recording:
            self.stop_recording()

        # kill model_loader
        if self.model_loader and self.model_loader.isRunning():
            print("Stopping model loader...")
            self.model_loader.terminate()

        # kill transcribing_thread
        if self.transcribing_thread and self.transcribing_thread.isRunning():
            print("Stopping transcribing thread...")
            self.transcribing_thread.terminate()
            self.transcribing_thread = None

        winsound.PlaySound(None, winsound.SND_PURGE)  # Stop any playing sounds
        if self.listener:
            self.listener.stop()
        if self.dialog:
            self.dialog.close()
        if self.tray:
            self.tray.setVisible(False)
        if self.check_timer:
            self.check_timer.stop()

        # check animation thread
        if self.loading_thread and self.loading_thread.isRunning():
            print("Stopping loading animation...")
            self.loading_thread.terminate()
            self.loading_thread = None

        self.app.quit()

    def run(self):
        return self.app.exec()

    def create_device_submenu(self, parent_menu):
        device_menu = QMenu("Select Device", parent_menu)
        # Create action group for radio buttons
        device_group = QActionGroup(device_menu)
        device_group.setExclusive(True)
        # Add GPU mode only if CUDA is available
        if self.cuda_device_count > 0:
            # Create CUDA devices submenu
            cuda_menu = QMenu("GPU (CUDA)", device_menu)
            cuda_group = QActionGroup(cuda_menu)
            cuda_group.setExclusive(True)
            for device_id in range(self.cuda_device_count):
                device_action = QAction(f"Device {device_id}", cuda_menu)
                device_action.setCheckable(True)
                device_action.setChecked(self.device_mode == "cuda" and self.cuda_device == device_id)
                device_action.triggered.connect(lambda checked, d=device_id: self.change_device_mode("cuda", cuda_device=d))
                cuda_group.addAction(device_action)
                cuda_menu.addAction(device_action)
            device_menu.addMenu(cuda_menu)
        # Add CPU mode
        cpu_action = QAction("CPU", device_menu)
        cpu_action.setCheckable(True)
        cpu_action.setChecked(self.device_mode == "cpu")
        cpu_action.triggered.connect(lambda checked: self.change_device_mode("cpu"))
        device_group.addAction(cpu_action)
        device_menu.addAction(cpu_action)
        # Store reference to menu to update enabled state
        self.device_menu = device_menu
        return device_menu

    def change_device_mode(self, device, cuda_device=0):
        # Don't allow device changes while loading
        if self.model_loader and self.model_loader.isRunning():
            print("Cannot change device while model is being loaded")
            return
        if device != self.device_mode or (device == "cuda" and cuda_device != self.cuda_device):
            self.device_mode = device
            if device == "cuda":
                self.cuda_device = cuda_device
            self.save_config()  # Save when device changes
            self.model = None  # Clear current model
            # Stop any existing loading animation
            if self.loading_thread and self.loading_thread.isRunning():
                self.loading_thread.stop()
                self.loading_thread = None
            # Create and start a new loading animation thread
            self.loading_thread = LoadingIconThread()
            self.loading_thread.update_icon.connect(self._update_tray_icon)
            self.loading_thread.start()
            self.load_whisper_model()

    def toggle_auto_paste(self, checked):
        self.auto_paste = checked
        self.save_config()


def get_models_directory():
    """Get the models directory in AppData/Local."""
    app_data = Path(os.getenv("LOCALAPPDATA"))
    models_dir = app_data / "VibeHotkeyWindows" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return str(models_dir)


if __name__ == "__main__":
    app = HotkeyApp()
    sys.exit(app.run())
