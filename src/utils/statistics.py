from collections import defaultdict
from typing import List


class GlobalStatistics:
    """
    A class to manage global statistics for the application.
    This class is designed
    """

    def __init__(self):
        self.model_tokens = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "request_cnt": 0}
        )
        self.agent_times: List[dict] = []

    def update_model_tokens(
        self, model_name: str, input_tokens: int, output_tokens: int
    ):
        """
        Update the token counts for a specific model.
        """
        self.model_tokens[model_name]["input_tokens"] += input_tokens
        self.model_tokens[model_name]["output_tokens"] += output_tokens
        self.model_tokens[model_name]["request_cnt"] += 1

    def add_time_entry(self, data: dict):
        """
        Update the time taken by an agent for a specific task.
        """
        self.agent_times.append(data)

    def get_agent_times(self):
        """
        Get the list of agent times.
        """
        return self.agent_times

    def get_model_tokens(self):
        """
        Get the token counts for all models.
        """
        return self.model_tokens

    def get_statistics(self):
        """
        Get a summary of the global statistics.
        """
        return {
            "model_tokens": self.get_model_tokens(),
            "agent_times": self.get_agent_times(),
        }


global_statistics = GlobalStatistics()

import functools
from datetime import datetime
import asyncio


def timed_step(step_name):
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(self, *args, **kwargs)
                return result
            finally:
                end_time = datetime.now()
                time_entry = {
                    "step_name": step_name + start_time.isoformat(),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                }
                global_statistics.add_time_entry(time_entry)

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                end_time = datetime.now()
                time_entry = {
                    "step_name": step_name + start_time.isoformat(),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                }
                global_statistics.add_time_entry(time_entry)

        # 判断是否为异步函数
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
