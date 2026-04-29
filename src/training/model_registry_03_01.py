"""
03-01 Model Registry: Candidate models for VeritasCarbon fine-tuning.

Maintains a registry of all candidate student models with their configurations,
hardware requirements, and loading utilities. Supports the Teacher→Student
knowledge distillation paradigm where Qwen2-72B is the teacher.

Naming convention: filename_03_01.py (notebook 03, 1st module)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model recommendation tier for the experiment plan."""
    PRIMARY = "primary"           # Main student (must do)
    SCALING = "scaling"           # Model scale ablation point
    SELF_TRAINING = "self_train"  # Self-training control
    CROSS_ARCH = "cross_arch"     # Cross-architecture generalization
    OPTIONAL = "optional"         # Nice-to-have


class Architecture(Enum):
    """Model architecture type."""
    DENSE = "dense"
    MOE = "moe"


# ═══════════════════════════════════════════════════════════════════
#  LOCAL MODEL STORAGE
# ═══════════════════════════════════════════════════════════════════

# Base directory for locally downloaded model weights.
# All models are stored as: MODEL_BASE_DIR / <folder_name> /
MODEL_BASE_DIR = Path("/hpc2hdd/home/yjiang909/models")


@dataclass
class ModelSpec:
    """Specification for a candidate fine-tuning model."""

    # ── Identity ──
    model_id: str                    # HuggingFace model ID
    display_name: str                # Human-readable name for tables
    short_name: str                  # Short key for file naming
    organization: str                # Model provider
    param_count: str                 # e.g., "7B", "14B"
    param_count_numeric: float       # In billions, for sorting

    # ── Architecture ──
    architecture: Architecture = Architecture.DENSE
    active_params: Optional[str] = None  # For MoE: active param count

    # ── Capabilities ──
    chinese_capability: str = "excellent"   # excellent / good / fair
    max_context_length: int = 8192
    license_type: str = "Apache 2.0"

    # ── Hardware ──
    qlora_vram_gb: float = 16.0       # Estimated VRAM for QLoRA
    min_gpus: int = 1                  # Minimum GPU count
    recommended_gpu: str = "A100-80G"

    # ── Experiment ──
    tier: ModelTier = ModelTier.PRIMARY
    experiment_role: str = ""          # Description of role in experiment
    rationale: str = ""                # Why this model is selected

    # ── Local path ──
    local_dir_name: str = ""           # Folder name under MODEL_BASE_DIR
    disk_size_gb: float = 0.0          # Approximate download size

    # ── QLoRA defaults (can be overridden per model) ──
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # ── Unsloth compatibility ──
    unsloth_compatible: bool = True  # False for models with known Unsloth bugs
    max_seq_length_override: int = 0  # Override max_seq_length for Unsloth (0 = use default)
    dtype_override: str = ""  # "float16" or "bfloat16" (empty = auto)

    # ── Training defaults ──
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def local_path(self) -> Optional[Path]:
        """Resolved local path if the model is downloaded."""
        if not self.local_dir_name:
            return None
        p = MODEL_BASE_DIR / self.local_dir_name
        return p if p.exists() else None

    @property
    def resolved_path(self) -> str:
        """Return local path if available, otherwise HuggingFace model ID."""
        lp = self.local_path
        if lp is not None:
            return str(lp)
        return self.model_id

    @property
    def is_downloaded(self) -> bool:
        """Check if model exists locally."""
        lp = self.local_path
        if lp is None:
            return False
        # Check for at least one safetensors or bin file
        return (any(lp.glob("*.safetensors"))
                or any(lp.glob("*.bin"))
                or any(lp.glob("model-*.safetensors")))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging and config export."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "short_name": self.short_name,
            "organization": self.organization,
            "param_count": self.param_count,
            "architecture": self.architecture.value,
            "tier": self.tier.value,
            "qlora_vram_gb": self.qlora_vram_gb,
            "disk_size_gb": self.disk_size_gb,
            "local_dir_name": self.local_dir_name,
            "is_downloaded": self.is_downloaded,
            "resolved_path": self.resolved_path,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "effective_batch_size": self.effective_batch_size,
        }


# ═══════════════════════════════════════════════════════════════════
#  MODEL REGISTRY — All candidate models for VeritasCarbon
# ═══════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    """Register a model spec in the global registry."""
    MODEL_REGISTRY[spec.short_name] = spec
    return spec


# ── Tier-1: Primary Experiments (Qwen family) ──

register_model(ModelSpec(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    display_name="Qwen2.5-7B",
    short_name="qwen2.5-7b",
    organization="Alibaba Cloud",
    param_count="7B",
    param_count_numeric=7.6,
    max_context_length=32768,
    license_type="Apache 2.0",
    qlora_vram_gb=16.0,
    min_gpus=1,
    recommended_gpu="A100-40G",
    tier=ModelTier.PRIMARY,
    experiment_role="Primary student model",
    rationale=(
        "Same model family as the 72B teacher (shared tokenizer & architecture), "
        "eliminating architecture confounds. 7B is the standard student size for "
        "knowledge distillation (LIMA, Alpaca, WizardLM). Lowest compute cost "
        "enables extensive hyperparameter search. Suggested paper narrative: "
        "'large model distills domain expertise into deployable small model'."
    ),
    local_dir_name="Qwen2.5-7B-Instruct",
    disk_size_gb=15.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
))

register_model(ModelSpec(
    model_id="Qwen/Qwen2.5-14B-Instruct",
    display_name="Qwen2.5-14B",
    short_name="qwen2.5-14b",
    organization="Alibaba Cloud",
    param_count="14B",
    param_count_numeric=14.2,
    max_context_length=32768,
    license_type="Apache 2.0",
    qlora_vram_gb=28.0,
    min_gpus=1,
    recommended_gpu="A100-80G",
    tier=ModelTier.SCALING,
    experiment_role="Scaling ablation mid-point",
    rationale=(
        "Critical mid-point for the model scale ablation: 7B → 14B → 72B. "
        "Enables plotting a proper scaling curve. 14B offers the best balance "
        "between capability and deployment cost for real-world ESG applications."
    ),
    local_dir_name="Qwen2.5-14B-Instruct",
    disk_size_gb=28.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
))

register_model(ModelSpec(
    model_id="Qwen/Qwen2-72B-Instruct",
    display_name="Qwen2-72B",
    short_name="qwen2-72b",
    organization="Alibaba Cloud",
    param_count="72B",
    param_count_numeric=72.7,
    max_context_length=32768,
    license_type="Tongyi Qianwen License",
    qlora_vram_gb=48.0,
    min_gpus=1,
    recommended_gpu="A100-80G",
    tier=ModelTier.SELF_TRAINING,
    experiment_role="Self-training control (teacher model itself)",
    rationale=(
        "Control experiment: fine-tuning the teacher model (Qwen2-72B) itself "
        "with CoDE data. Answers the reviewer question: 'Does the CoDE data "
        "improve an already-strong 72B model?' Also serves as upper bound for "
        "the scaling ablation. Same model used for QA generation."
    ),
    local_dir_name="Qwen2-72B-Instruct/Qwen/Qwen2-72B-Instruct",
    disk_size_gb=137.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,  # Lower LR for large model
    max_seq_length_override=1024,  # Reduce from 2048 to fit in 80GB single GPU
    dtype_override="float16",  # Explicit fp16 to reduce memory overhead
))

# ── Tier-2: Cross-Architecture Generalization ──

register_model(ModelSpec(
    model_id="THUDM/glm-4-9b-chat",
    display_name="GLM-4-9B",
    short_name="glm4-9b",
    organization="Zhipu AI / Tsinghua",
    param_count="9B",
    param_count_numeric=9.0,
    max_context_length=131072,
    license_type="Zhipu Open License",
    qlora_vram_gb=20.0,
    min_gpus=1,
    recommended_gpu="A100-40G",
    tier=ModelTier.CROSS_ARCH,
    experiment_role="Cross-architecture: GLM family",
    rationale=(
        "Project already uses Zhipu GLM-4 API for QA generation (config api.provider='zhipu'). "
        "Cross-family validation proves CoDE data generalizes beyond Qwen. "
        "Tsinghua-backed model has high academic credibility in Chinese NLP community."
    ),
    local_dir_name="glm-4-9b-chat",
    disk_size_gb=18.0,
    lora_target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    unsloth_compatible=False,  # Unsloth's ChatGLM patch: 'NoneType' has no attr 'weight'
))

register_model(ModelSpec(
    model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
    display_name="DeepSeek-V2-Lite",
    short_name="deepseek-v2-lite",
    organization="DeepSeek",
    param_count="16B (2.4B active)",
    param_count_numeric=16.0,
    architecture=Architecture.MOE,
    active_params="2.4B",
    max_context_length=32768,
    license_type="DeepSeek License",
    qlora_vram_gb=24.0,
    min_gpus=1,
    recommended_gpu="A100-80G",
    tier=ModelTier.CROSS_ARCH,
    experiment_role="Cross-architecture: MoE representative",
    rationale=(
        "Only MoE architecture in the candidate list — provides architecture diversity. "
        "16B total params but only 2.4B active → extremely fast inference, ideal for "
        "deployment. DeepSeek excels in financial/analytical domains, aligning with ESG."
    ),
    local_dir_name="DeepSeek-V2-Lite-Chat",
    disk_size_gb=32.0,
    # MLA attention only — do NOT include gate_proj/up_proj/down_proj
    # as they exist inside each of the 64 MoE experts (5184 LoRA adapters → OOM).
    # q_lora_rank=None → uses q_proj directly; kv uses compressed MLA path.
    lora_target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,        # Same as all other models for fair comparison
    unsloth_compatible=False,  # DeepSeekV2 forward() rejects Unsloth's causal_mask kwarg
))

register_model(ModelSpec(
    model_id="internlm/internlm2_5-7b-chat",
    display_name="InternLM2.5-7B",
    short_name="internlm2.5-7b",
    organization="Shanghai AI Lab",
    param_count="7B",
    param_count_numeric=7.0,
    max_context_length=1048576,  # 1M context
    license_type="Apache 2.0",
    qlora_vram_gb=16.0,
    min_gpus=1,
    recommended_gpu="A100-40G",
    tier=ModelTier.CROSS_ARCH,
    experiment_role="Cross-architecture: InternLM family",
    rationale=(
        "Outstanding long-context capability (1M tokens) — ESG reports are long documents. "
        "Strong agent/tool-use abilities align with CoDE's multi-expert design. "
        "Shanghai AI Lab (national team) has high academic credibility."
    ),
    local_dir_name="internlm2_5-7b-chat",
    disk_size_gb=15.0,
    lora_target_modules=["wqkv", "wo", "w1", "w2", "w3"],
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    unsloth_compatible=False,  # InternLM2 forward() rejects Unsloth's causal_mask kwarg
))

# ── Tier-3: Optional ──

register_model(ModelSpec(
    model_id="01-ai/Yi-1.5-9B-Chat",
    display_name="Yi-1.5-9B",
    short_name="yi1.5-9b",
    organization="01.AI",
    param_count="9B",
    param_count_numeric=9.0,
    max_context_length=4096,
    license_type="Apache 2.0",
    qlora_vram_gb=18.0,
    min_gpus=1,
    recommended_gpu="A100-40G",
    tier=ModelTier.OPTIONAL,
    experiment_role="Optional: additional cross-family validation",
    rationale=(
        "Yi series has strong Chinese capability with bilingual training. "
        "9B size provides comparison with GLM-4-9B at similar scale."
    ),
    local_dir_name="Yi-1.5-9B-Chat",
    disk_size_gb=18.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
))


# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_model(short_name: str) -> ModelSpec:
    """Get a model spec by short name."""
    if short_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{short_name}' not found. Available: {available}")
    return MODEL_REGISTRY[short_name]


def get_models_by_tier(tier: ModelTier) -> List[ModelSpec]:
    """Get all models for a given tier."""
    return [m for m in MODEL_REGISTRY.values() if m.tier == tier]


def get_experiment_plan(
    include_tiers: Optional[List[ModelTier]] = None,
) -> List[ModelSpec]:
    """Get the ordered experiment plan.

    Default: PRIMARY + SCALING + SELF_TRAINING + CROSS_ARCH
    """
    if include_tiers is None:
        include_tiers = [
            ModelTier.PRIMARY,
            ModelTier.SCALING,
            ModelTier.SELF_TRAINING,
            ModelTier.CROSS_ARCH,
        ]
    models = [m for m in MODEL_REGISTRY.values() if m.tier in include_tiers]
    # Sort: PRIMARY first, then by param count
    tier_order = {t: i for i, t in enumerate(include_tiers)}
    models.sort(key=lambda m: (tier_order.get(m.tier, 99), m.param_count_numeric))
    return models


def print_registry_summary() -> None:
    """Print a formatted summary of all registered models."""
    tier_labels = {
        ModelTier.PRIMARY: "Tier-1: Primary Student",
        ModelTier.SCALING: "Tier-1: Scaling Ablation",
        ModelTier.SELF_TRAINING: "Tier-1: Self-Training Control",
        ModelTier.CROSS_ARCH: "Tier-2: Cross-Architecture",
        ModelTier.OPTIONAL: "Tier-3: Optional",
    }

    print("=" * 90)
    print("  VeritasCarbon Model Registry — Candidate Fine-tuning Models")
    print("=" * 90)
    print(f"\n  Teacher model: Qwen2-72B-Instruct (used for CoDE QA generation)")
    print(f"  Training data: train_filtered.jsonl (31,315 QA pairs, Alpaca format)")
    print(f"  Model storage: {MODEL_BASE_DIR}\n")

    current_tier = None
    for model in sorted(MODEL_REGISTRY.values(),
                        key=lambda m: (list(ModelTier).index(m.tier), m.param_count_numeric)):
        if model.tier != current_tier:
            current_tier = model.tier
            print(f"\n{'─' * 90}")
            print(f"  {tier_labels.get(current_tier, current_tier.value)}")
            print(f"{'─' * 90}")

        arch_str = f" ({model.active_params} active)" if model.active_params else ""
        dl_status = "✅ Downloaded" if model.is_downloaded else "❌ Not downloaded"
        print(f"\n  [{model.short_name}] {model.display_name}")
        print(f"    Model ID:      {model.model_id}")
        print(f"    Organization:  {model.organization}")
        print(f"    Params:        {model.param_count}{arch_str}  |  Arch: {model.architecture.value}")
        print(f"    VRAM (QLoRA):  ~{model.qlora_vram_gb:.0f}GB  |  GPU: {model.min_gpus}× {model.recommended_gpu}")
        print(f"    Disk size:     ~{model.disk_size_gb:.0f}GB  |  {dl_status}")
        print(f"    Local path:    {model.resolved_path}")
        print(f"    License:       {model.license_type}")
        print(f"    Role:          {model.experiment_role}")


def check_downloads() -> Dict[str, bool]:
    """Check which models are downloaded locally.

    Returns:
        Dict mapping short_name → is_downloaded.
    """
    status = {}
    total_needed = 0.0
    total_downloaded = 0.0

    print(f"Model download status ({MODEL_BASE_DIR}):")
    print(f"{'─' * 75}")
    print(f"  {'Model':<25} {'Size':>8} {'Status':>15} {'Local Path'}")
    print(f"{'─' * 75}")

    for model in sorted(MODEL_REGISTRY.values(),
                        key=lambda m: (list(ModelTier).index(m.tier), m.param_count_numeric)):
        dl = model.is_downloaded
        status[model.short_name] = dl
        icon = "✅" if dl else "❌"
        path_str = str(model.local_path) if dl else "(not found)"
        print(f"  {model.short_name:<25} {model.disk_size_gb:>6.0f}GB  {icon:>12}  {path_str}")
        total_needed += model.disk_size_gb
        if dl:
            total_downloaded += model.disk_size_gb

    n_ok = sum(1 for v in status.values() if v)
    n_total = len(status)
    missing_gb = total_needed - total_downloaded
    print(f"{'─' * 75}")
    print(f"  Downloaded: {n_ok}/{n_total} models "
          f"({total_downloaded:.0f}GB / {total_needed:.0f}GB)")
    if missing_gb > 0:
        print(f"  Need to download: ~{missing_gb:.0f}GB more")
    else:
        print(f"  All models ready! ✅")

    return status
