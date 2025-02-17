from typing import List
from pathlib import Path
import sys
import os
import ctranslate2


def set_cuda_paths() -> None:
    """
    Taken from https://github.com/SYSTRAN/faster-whisper/issues/1080#issuecomment-2429688038
    This fixes all path related issues with CUDA for me during dev.
    Originally written by https://github.com/BBC-Esq
    """
    venv_base: Path = Path(sys.executable).parent.parent
    nvidia_base_path: Path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path: Path = nvidia_base_path / 'cuda_runtime' / 'bin'
    cublas_path: Path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path: Path = nvidia_base_path / 'cudnn' / 'bin'
    paths_to_add: List[str] = [str(cuda_path), str(cublas_path), str(cudnn_path)]
    env_vars: List[str] = ['CUDA_PATH', 'CUDA_PATH_V12_4', 'PATH']
    
    for env_var in env_vars:
        # print(f"Setting {env_var} to {paths_to_add}")
        current_value: str = os.environ.get(env_var, '')
        new_value: str = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
        os.environ[env_var] = new_value


def check_cuda_availability() -> int:
    """Check if CUDA is available and return number of devices."""
    try:
        device_count: int = ctranslate2.get_cuda_device_count()
        if device_count == 0:
            print("No CUDA devices found")
            return 0
        return device_count
    except RuntimeError as e:
        print(f"CUDA runtime error: {e}")
        return 0
