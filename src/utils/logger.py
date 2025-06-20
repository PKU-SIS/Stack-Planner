from pathlib import Path
import sys
from datetime import datetime
import inspect
from enum import Enum
from typing import Union
from loguru import logger as _logger

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()


_print_level = "INFO"


class Logger:
    def __init__(self, name: str = None, log_level: str = "INFO"):
        self.logger = define_log_level(name=name,print_level=log_level,logfile_level=log_level)
        
    def set_log_level(self, log_level: str = "DEBUG"):
        """Set the log level for both console and file outputs
        Args:
            print_level: The log level for console output
            logfile_level: The log level for file output
        """
        self.logger = define_log_level(
            print_level=log_level.upper(),
            logfile_level=log_level.upper(),
            name=self.logger.name if hasattr(self.logger, 'name') else None
        )

    def _log_one_line(self, level, content):
        # Get the frame two levels up in the call stack (skipping this method and the calling log method)
        frame = inspect.currentframe().f_back.f_back.f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        content = content.replace('\n', '2f2f2f')
        formatted_content = f"{filename}:{lineno}|{content}"
        getattr(self.logger, level)(formatted_content)
    
    def _log(self, level, content):    
        try: 
            content = str(content).encode('utf-8', errors='replace').decode('utf-8')
        except Exception as e:
            print(f"Error encoding log content")
            return
        
        if isinstance(content,str) and len(content)>10000:
            chunks = [content[i:i + 10000] for i in range(0, len(content), 10000)]
            for i, chunk in enumerate(chunks):
                self._log_one_line(level, f"llm_content|part_{i + 1}/{len(chunks)}|{chunk}")
        else:
            self._log_one_line(level, content)

            
    def info(self, content):        
        self._log("info", content)


    def debug(self, content):
        self._log("debug", content)

    def warning(self, content):
        self._log("warning", content)

    def error(self, content):
        self._log("error", content)

    def exception(self, content):
        self._log("exception", content)

    def save_llm_content(self, input):
        if len(input) <= 8000:
            self._log("info", f"llm_content|{input}")
        else:
            chunk_size = 8000
            total_chunks = (len(input) + chunk_size - 1) // chunk_size  # 向上取整
            for i in range(0, len(input), chunk_size):
                chunk = input[i:i + chunk_size]
                chunk_number = i // chunk_size + 1
                self._log("info", f"llm_content|part_{chunk_number}/{total_chunks}|{chunk}")

def define_log_level(print_level="INFO", logfile_level="INFO", name: str = None):
    """Adjust the log level to above level"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )  # name a log with prefix name

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    try:
        _logger.add("/root/logs/job.log", level=logfile_level)
    except Exception as e:
        _logger.add(PROJECT_ROOT / f"logs/{log_name}.log", level=logfile_level)
    return _logger


logger = Logger()
if __name__ == "__main__":

    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
