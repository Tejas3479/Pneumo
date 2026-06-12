import pytest
import torch
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

    # Mock ResNet-50 to load without weights/pretrained (offline-safe)
    import torchvision.models as models
    original_resnet50 = models.resnet50
    def mocked_resnet50(*args, **kwargs):
        kwargs["weights"] = None
        if "pretrained" in kwargs:
            kwargs["pretrained"] = False
        return original_resnet50(*args, **kwargs)
    monkeypatch.setattr(models, "resnet50", mocked_resnet50)

    # Mock Hugging Face ViT Config and Model to load tiny configurations (offline-safe & fast)
    from transformers import ViTConfig, ViTForImageClassification
    def mocked_vit_config_from_pretrained(*args, **kwargs):
        cfg = ViTConfig(
            num_labels=1,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            output_hidden_states=True,
            output_attentions=True
        )
        return cfg

    def mocked_vit_from_pretrained(*args, **kwargs):
        cfg = kwargs.get("config")
        if cfg is None:
            cfg = mocked_vit_config_from_pretrained()
        return ViTForImageClassification(cfg)

    monkeypatch.setattr(ViTConfig, "from_pretrained", mocked_vit_config_from_pretrained)
    monkeypatch.setattr(ViTForImageClassification, "from_pretrained", mocked_vit_from_pretrained)

    # Mock load_from_checkpoint to prevent loading large/incompatible disk checkpoints during testing
    from src.model_foundation import ViTPneumothoraxClassifier
    from src.model import PneumothoraxClassifier
    from src.model_medfound import MedicalFoundationClassifier
    monkeypatch.setattr(ViTPneumothoraxClassifier, "load_from_checkpoint", lambda ckpt_path, *args, **kwargs: ViTPneumothoraxClassifier())
    monkeypatch.setattr(PneumothoraxClassifier, "load_from_checkpoint", lambda ckpt_path, *args, **kwargs: PneumothoraxClassifier())
    monkeypatch.setattr(MedicalFoundationClassifier, "load_from_checkpoint", lambda ckpt_path, *args, **kwargs: MedicalFoundationClassifier())

    # Mock AutoModel and CLIPModel for offline medical foundation model testing
    class DummyVisionModelOutput:
        def __init__(self, last_hidden_state, pooler_output):
            self.last_hidden_state = last_hidden_state
            self.pooler_output = pooler_output

    class DummyConfig:
        hidden_size = 768

    class DummyVisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig()
            self.q_proj = torch.nn.Linear(768, 768)
            self.v_proj = torch.nn.Linear(768, 768)
        def forward(self, x, *args, **kwargs):
            batch_size = x.shape[0]
            last_hidden_state = torch.ones(batch_size, 197, 768) * 0.5
            pooler_output = torch.ones(batch_size, 768) * 0.5
            return DummyVisionModelOutput(last_hidden_state, pooler_output)

    class DummyCLIPModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = DummyVisionModel()
        def forward(self, x, *args, **kwargs):
            return self.vision_model(x)

    from transformers import AutoModel, CLIPModel
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: DummyVisionModel())
    monkeypatch.setattr(CLIPModel, "from_pretrained", lambda *args, **kwargs: DummyCLIPModel())
