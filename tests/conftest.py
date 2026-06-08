import pytest
from unittest.mock import MagicMock
from celery.app.task import Task
from celery.result import AsyncResult

# Dictionary to hold results of eagerly executed tasks
_eager_results = {}

# Store original methods
original_apply_async = Task.apply_async
original_async_result = AsyncResult

def patched_apply_async(self, args=None, kwargs=None, *args_rest, **kwargs_rest):
    # Call original apply_async (which executes the task synchronously in eager mode)
    result = original_apply_async(self, args, kwargs, *args_rest, **kwargs_rest)
    if result and hasattr(result, 'id'):
        _eager_results[result.id] = result
    return result

class PatchedAsyncResult:
    def __init__(self, id, *args, **kwargs):
        self.id = id
        if id in _eager_results:
            eager_res = _eager_results[id]
            self.state = eager_res.state
            self.result = eager_res.result
            self.info = eager_res.info
        else:
            self.state = "PENDING"
            self.result = None
            self.info = None

@pytest.fixture(autouse=True)
def setup_celery_eager_and_mocks(monkeypatch):
    from app.tasks import celery_app
    celery_app.conf.update(task_always_eager=True)
    
    # Apply patches
    monkeypatch.setattr(Task, "apply_async", patched_apply_async)
    monkeypatch.setattr("app.main.AsyncResult", PatchedAsyncResult)
