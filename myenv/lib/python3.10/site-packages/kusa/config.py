import os
from dotenv import load_dotenv, find_dotenv
from typing import Optional
import warnings

class Config:
    """
    Configuration handler with automatic .env reloading and change detection.
    """
    
    _last_modified_time = 0
    _env_path = None
    
    @classmethod
    def _locate_env_file(cls):
        """Find and cache the .env file location"""
        if cls._env_path is None:
            cls._env_path = find_dotenv(usecwd=True)
        return cls._env_path
    
    @classmethod
    def _should_reload(cls):
        """Check if .env file has been modified"""
        if not os.path.exists(cls._locate_env_file()):
            return False
            
        current_mtime = os.path.getmtime(cls._locate_env_file())
        if current_mtime > cls._last_modified_time:
            cls._last_modified_time = current_mtime
            return True
        return False
    
    @classmethod
    def reload_env(cls, force=False):
        """Reload .env file if it has changed or when forced"""
        if force or cls._should_reload():
            env_file = cls._locate_env_file()
            if os.path.exists(env_file):
                load_dotenv(env_file, override=True)
            else:
                warnings.warn(f"No .env file found at {env_file}")

    @staticmethod
    def get_base_url() -> str:
        """Get BASE_URL with automatic change detection"""
        Config.reload_env()
        base_url = os.getenv("BASE_URL")
        # base_url = "https://kusa.zadulmead.org/dataset"
                
        if not base_url:
            raise ValueError("BASE_URL must be set in .env file or environment")
        return base_url.rstrip('/')

    @staticmethod
    def get_encryption_key() -> str:
        """Get ENCRYPTION_KEY with automatic change detection"""
        Config.reload_env()
        key = os.getenv("SECRET_KEY")
        key = key[:32] if key else None

        if not key:
            raise ValueError("ENCRYPTION_KEY must be set in .env file or environment")
        return key