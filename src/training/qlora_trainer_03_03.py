"""
03-03 QLoRA Trainer: Fine-tuning pipeline with QLoRA for VeritasCarbon.

Wraps HuggingFace Trainer with QLoRA (4-bit quantization + LoRA adapters),
supporting multi-model experiments, checkpointing, and result logging.

Naming convention: filename_03_03.py (notebook 03, 3rd module)

Dependencies:
    pip install transformers>=4.40.0 peft>=0.10.0 bitsandbytes>=0.43.0
    pip install trl>=0.8.0 datasets accelerate
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  UNSLOTH DETECTION (lazy — avoids importing unsloth at module load)
# ═══════════════════════════════════════════════════════════════════

_unsloth_cached: Optional[bool] = None


def _check_unsloth() -> bool:
    """Check if unsloth is available (lazy, cached)."""
    global _unsloth_cached
    if _unsloth_cached is not None:
        return _unsloth_cached
    try:
        import unsloth  # noqa: F401
        _unsloth_cached = True
    except ImportError:
        _unsloth_cached = False
    return _unsloth_cached


class _LazyUnsloth:
    """Lazy proxy that only imports unsloth when accessed as a bool."""
    def __bool__(self):
        return _check_unsloth()
    def __repr__(self):
        return f"UNSLOTH_AVAILABLE({_check_unsloth()})"


UNSLOTH_AVAILABLE = _LazyUnsloth()


@dataclass
class TrainingResult:
    """Stores the result of a single fine-tuning run."""
    model_short_name: str
    model_id: str
    param_count: str
    # Training info
    num_train_samples: int = 0
    num_eval_samples: int = 0
    num_epochs: float = 0.0
    total_steps: int = 0
    training_time_seconds: float = 0.0
    # Final metrics
    train_loss: float = 0.0
    eval_loss: float = 0.0
    # Paths
    output_dir: str = ""
    adapter_path: str = ""
    # Config snapshot
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    effective_batch_size: int = 16
    max_length: int = 2048
    # Method
    training_method: str = "qlora"  # qlora / full_ft / unsloth_qlora
    use_unsloth: bool = False
    # Status
    status: str = "pending"  # pending / running / completed / failed
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print_summary(self) -> None:
        hrs = self.training_time_seconds / 3600
        print(f"\n{'═' * 60}")
        print(f"  Training Result: {self.model_short_name}")
        print(f"{'═' * 60}")
        print(f"  Model:          {self.model_id}")
        print(f"  Params:         {self.param_count}")
        print(f"  Status:         {self.status}")
        print(f"  Train samples:  {self.num_train_samples:,}")
        print(f"  Eval samples:   {self.num_eval_samples:,}")
        print(f"  Epochs:         {self.num_epochs}")
        print(f"  Total steps:    {self.total_steps:,}")
        print(f"  Train loss:     {self.train_loss:.4f}")
        print(f"  Eval loss:      {self.eval_loss:.4f}")
        print(f"  Time:           {hrs:.2f} hours")
        print(f"  Adapter:        {self.adapter_path}")
        print(f"{'═' * 60}")


# ═══════════════════════════════════════════════════════════════════
#  QLORA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

def get_bnb_config():
    """Get BitsAndBytes 4-bit quantization config."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    task_type: str = "CAUSAL_LM",
):
    """Get LoRA adapter config.

    Args:
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout.
        target_modules: List of module names to apply LoRA to.
        task_type: Task type for PEFT.

    Returns:
        LoraConfig object.
    """
    from peft import LoraConfig, TaskType

    task_type_enum = TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else task_type

    if target_modules is None:
        # Default: Qwen-style modules
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type_enum,
    )


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_id: str,
    max_length: int = 2048,
    use_qlora: bool = True,
    use_unsloth: bool = False,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = "flash_attention_2",
    dtype_override: Optional[str] = None,
):
    """Load a model and tokenizer with optional QLoRA quantization + Unsloth.

    Args:
        model_id: HuggingFace model ID or local path.
        max_length: Maximum sequence length.
        use_qlora: Whether to load with 4-bit quantization.
        use_unsloth: Whether to use Unsloth for 2-5x faster training.
        device_map: Device mapping strategy.
        trust_remote_code: Whether to trust remote code.
        attn_implementation: Attention implementation (flash_attention_2, sdpa, eager).
        dtype_override: Explicit dtype string ("float16" or "bfloat16"). None = auto.

    Returns:
        (model, tokenizer) tuple.  If use_unsloth=True, model is an Unsloth-patched model.
    """
    import torch

    # Resolve dtype override
    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    resolved_dtype = _dtype_map.get(dtype_override) if dtype_override else None

    # ── Unsloth path (preferred when available) ──────────────────
    if use_unsloth:
        if not UNSLOTH_AVAILABLE:
            logger.warning("Unsloth requested but not installed. Falling back to HF.")
        else:
            from unsloth import FastLanguageModel
            logger.info(f"Loading with Unsloth: {model_id} (4bit={use_qlora}, "
                        f"seq_len={max_length}, dtype={resolved_dtype})")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_length,
                dtype=resolved_dtype,  # None = auto, or explicit fp16/bf16
                load_in_4bit=use_qlora,
                trust_remote_code=trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Unsloth model loaded: {n_params / 1e9:.2f}B params")
            return model, tokenizer

    # ── Standard HuggingFace path ────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_id} (QLoRA={use_qlora})")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        model_max_length=max_length,
        padding_side="right",
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }

    if use_qlora:
        model_kwargs["quantization_config"] = get_bnb_config()
        # Force non-quantized layers to bf16 so Unsloth's patched trainer
        # does not complain about fp16/bf16 mismatch.
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        # Full fine-tuning in bf16
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Try flash attention, fall back if unavailable
    if attn_implementation:
        try:
            model_kwargs["attn_implementation"] = attn_implementation
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except (ImportError, ValueError) as e:
            logger.warning(f"Flash attention unavailable: {e}. Falling back to sdpa.")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.config.use_cache = False  # Disable KV cache for training

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {n_params / 1e9:.2f}B params, "
                f"{n_trainable / 1e6:.2f}M trainable (before LoRA)")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_finetune(
    model_spec,
    train_dataset,
    eval_dataset,
    output_base_dir: str = "models/finetuned",
    max_length: int = 2048,
    template_name: str = "alpaca_en",
    num_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    warmup_steps: int = 100,
    report_to: str = "tensorboard",
    use_unsloth: bool = False,
    full_finetune: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    dry_run: bool = False,
) -> TrainingResult:
    """Run QLoRA fine-tuning for a single model.

    Args:
        model_spec: ModelSpec from model_registry_03_01.
        train_dataset: HuggingFace Dataset (tokenized).
        eval_dataset: HuggingFace Dataset (tokenized).
        output_base_dir: Base directory for saving adapters.
        max_length: Max sequence length.
        template_name: Alpaca template variant.
        num_epochs: Override model_spec epochs.
        learning_rate: Override model_spec LR.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        warmup_steps: Warmup steps.
        report_to: Reporting backend ('tensorboard', 'wandb', 'none').
        use_unsloth: Use Unsloth acceleration (2-5x faster, 30-50% less VRAM).
        full_finetune: Full fine-tuning instead of QLoRA (needs more VRAM).
        resume_from_checkpoint: Path to resume from.
        dry_run: If True, only setup but don't train.

    Returns:
        TrainingResult object.
    """
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from trl import SFTTrainer

    # Determine training method label
    if full_finetune:
        method = "full_ft"
    elif use_unsloth and UNSLOTH_AVAILABLE:
        method = "unsloth_qlora"
    else:
        method = "qlora"

    # Output dir includes method suffix for ablation runs
    dir_suffix = f"-{method}" if full_finetune else ""
    result = TrainingResult(
        model_short_name=model_spec.short_name + (f"-fullft" if full_finetune else ""),
        model_id=model_spec.model_id,
        param_count=model_spec.param_count,
        num_train_samples=len(train_dataset),
        num_eval_samples=len(eval_dataset),
        lora_r=0 if full_finetune else model_spec.lora_r,
        lora_alpha=0 if full_finetune else model_spec.lora_alpha,
        learning_rate=learning_rate or model_spec.learning_rate,
        effective_batch_size=model_spec.effective_batch_size,
        max_length=max_length,
        training_method=method,
        use_unsloth=(use_unsloth and UNSLOTH_AVAILABLE),
    )

    output_dir = Path(output_base_dir) / (model_spec.short_name + dir_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.output_dir = str(output_dir)
    result.adapter_path = str(output_dir / "adapter") if not full_finetune else str(output_dir)

    epochs = num_epochs or model_spec.num_train_epochs
    lr = learning_rate or model_spec.learning_rate
    result.num_epochs = epochs

    logger.info(f"{'=' * 60}")
    logger.info(f"  Starting fine-tuning: {model_spec.display_name}")
    logger.info(f"  Model:  {model_spec.model_id}")
    logger.info(f"  LoRA:   r={model_spec.lora_r}, α={model_spec.lora_alpha}")
    logger.info(f"  LR:     {lr}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch:  {model_spec.effective_batch_size} effective")
    logger.info(f"  Method: {method}")
    logger.info(f"  Unsloth: {use_unsloth and UNSLOTH_AVAILABLE}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'=' * 60}")

    if dry_run:
        result.status = "dry_run"
        logger.info("Dry run mode — skipping actual training.")
        return result

    try:
        result.status = "running"
        start_time = time.time()

        # 1. Load model & tokenizer (use local path if available)
        #    Key: use Unsloth for Full FT too, to avoid monkey-patch conflict.
        #    Unsloth patches Qwen2Attention globally — if we loaded QLoRA via
        #    Unsloth earlier in this process, a subsequent standard-HF load
        #    of the same architecture would crash (missing apply_qkv etc.).
        #    Fix: always go through Unsloth when it's requested, just with
        #    load_in_4bit=False for Full FT.
        model_path = model_spec.resolved_path
        logger.info(f"Loading from: {model_path}")
        use_qlora_flag = not full_finetune

        # Respect unsloth_compatible flag from model_spec
        actual_use_unsloth = use_unsloth and getattr(model_spec, 'unsloth_compatible', True)
        if use_unsloth and not actual_use_unsloth:
            logger.info(f"Model {model_spec.short_name} is not Unsloth-compatible, "
                        f"using standard HF loading instead.")

        # Respect max_seq_length_override and dtype_override from model_spec
        actual_max_length = getattr(model_spec, 'max_seq_length_override', 0) or max_length
        actual_dtype = getattr(model_spec, 'dtype_override', '') or None

        model, tokenizer = load_model_and_tokenizer(
            model_path,
            max_length=actual_max_length,
            use_qlora=use_qlora_flag,
            use_unsloth=actual_use_unsloth,  # keep Unsloth even for Full FT (if compatible)
            dtype_override=actual_dtype,
        )

        if full_finetune:
            # Full fine-tuning: all parameters trainable, no LoRA
            model.enable_input_require_grads()
            logger.info("Full fine-tuning mode: all parameters trainable.")
        elif actual_use_unsloth and UNSLOTH_AVAILABLE:
            # Unsloth LoRA: use Unsloth's optimized get_peft_model
            from unsloth import FastLanguageModel
            model = FastLanguageModel.get_peft_model(
                model,
                r=model_spec.lora_r,
                lora_alpha=model_spec.lora_alpha,
                lora_dropout=model_spec.lora_dropout,
                target_modules=model_spec.lora_target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
            logger.info("Unsloth LoRA applied.")
        else:
            # Standard QLoRA: prepare + apply LoRA
            from peft import get_peft_model, prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
            lora_config = get_lora_config(
                r=model_spec.lora_r,
                alpha=model_spec.lora_alpha,
                dropout=model_spec.lora_dropout,
                target_modules=model_spec.lora_target_modules,
            )
            model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        pct = trainable_params / total_params * 100 if total_params > 0 else 0
        logger.info(f"{'Full FT' if full_finetune else 'LoRA'} applied: "
                     f"{trainable_params / 1e6:.2f}M trainable / "
                     f"{total_params / 1e9:.2f}B total ({pct:.2f}%)")

        # 3b. Auto-detect checkpoint for resume
        if resume_from_checkpoint is None:
            ckpt_dirs = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
            if ckpt_dirs:
                resume_from_checkpoint = str(ckpt_dirs[-1])
                logger.info(f"Auto-resuming from: {resume_from_checkpoint}")

        # 4. Training arguments
        #    If model loaded with dtype_override="float16", must use fp16=True
        #    instead of bf16=True to match Unsloth's expectation.
        use_fp16 = (actual_dtype == "float16")
        optim = "adamw_torch" if full_finetune else "paged_adamw_8bit"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=model_spec.per_device_train_batch_size,
            gradient_accumulation_steps=model_spec.gradient_accumulation_steps,
            learning_rate=lr,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            logging_dir=str(output_dir / "tb_logs"),
            save_steps=save_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=not use_fp16,
            fp16=use_fp16,
            # Full FT needs standard grad-ckpt; QLoRA+Unsloth uses its own
            gradient_checkpointing=(
                True if full_finetune
                else (not (actual_use_unsloth and UNSLOTH_AVAILABLE))
            ),
            optim=optim,
            lr_scheduler_type="cosine",
            report_to=report_to,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        # 5. Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_length,
        )

        # 6. Trainer
        #    trl >= 0.20 renamed 'tokenizer' → 'processing_class'.
        #    Unsloth patches SFTTrainer to accept 'tokenizer', but the clean
        #    (no-Unsloth) path needs 'processing_class'.
        _trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        # Use the API that matches the trl version
        import inspect
        _sft_params = inspect.signature(SFTTrainer.__init__).parameters
        if 'processing_class' in _sft_params:
            _trainer_kwargs['processing_class'] = tokenizer
        else:
            _trainer_kwargs['tokenizer'] = tokenizer
        trainer = SFTTrainer(**_trainer_kwargs)

        # 7. Train
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        # 8. Save model / adapter
        if full_finetune:
            save_path = output_dir
            model.save_pretrained(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            logger.info(f"Full model saved to: {save_path}")
        else:
            adapter_path = output_dir / "adapter"
            model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))
            logger.info(f"Adapter saved to: {adapter_path}")

        # 9. Record results
        result.total_steps = trainer.state.global_step
        result.train_loss = train_result.training_loss
        result.training_time_seconds = time.time() - start_time

        # Evaluate
        eval_result = trainer.evaluate()
        result.eval_loss = eval_result.get("eval_loss", 0.0)

        result.status = "completed"

        # Save result metadata
        with open(output_dir / "training_result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    except Exception as e:
        result.status = "failed"
        result.error_message = str(e)
        logger.error(f"Training failed for {model_spec.short_name}: {e}")
        raise

    return result


# ═══════════════════════════════════════════════════════════════════
#  MULTI-MODEL EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_experiment_suite(
    model_specs: list,
    train_records: list,
    eval_records: list,
    output_base_dir: str = "models/finetuned",
    max_length: int = 2048,
    template_name: str = "alpaca_en",
    use_unsloth: bool = False,
    report_to: str = "tensorboard",
    dry_run: bool = False,
    skip_completed: bool = True,
) -> List[TrainingResult]:
    """Run fine-tuning for multiple models sequentially.

    Args:
        model_specs: List of ModelSpec objects.
        train_records: Training data (raw Alpaca records).
        eval_records: Evaluation data (raw Alpaca records).
        output_base_dir: Base output directory.
        max_length: Max sequence length.
        template_name: Alpaca template variant.
        use_unsloth: Use Unsloth acceleration (2-5x faster).
        report_to: Reporting backend ('tensorboard', 'wandb', 'none').
        dry_run: If True, simulation only.
        skip_completed: Skip models that already have training_result.json.

    Returns:
        List of TrainingResult objects.
    """
    from .data_loader_03_02 import prepare_tokenized_dataset

    results = []
    total = len(model_specs)

    print(f"\n{'═' * 70}")
    print(f"  VeritasCarbon Multi-Model Fine-tuning Suite")
    print(f"  Models: {total}")
    print(f"  Train:  {len(train_records):,} records")
    print(f"  Eval:   {len(eval_records):,} records")
    print(f"  Unsloth: {use_unsloth and UNSLOTH_AVAILABLE}")
    print(f"  Report:  {report_to}")
    print(f"  Dry run: {dry_run}")
    print(f"{'═' * 70}\n")

    for i, spec in enumerate(model_specs, 1):
        print(f"\n[{i}/{total}] {spec.display_name} ({spec.param_count}) "
              f"— {spec.experiment_role}")
        print(f"  Model ID: {spec.model_id}")

        # Check if already completed
        result_file = Path(output_base_dir) / spec.short_name / "training_result.json"
        if skip_completed and result_file.exists():
            with open(result_file) as f:
                prev = json.load(f)
            if prev.get("status") == "completed":
                print(f"  ⏭  Already completed. Skipping.")
                result = TrainingResult(**{
                    k: v for k, v in prev.items()
                    if k in TrainingResult.__dataclass_fields__
                })
                results.append(result)
                continue

        try:
            # Auto-detect checkpoint for resume (partial training)
            model_output_dir = Path(output_base_dir) / spec.short_name
            resume_ckpt = None
            ckpt_dirs = sorted(model_output_dir.glob("checkpoint-*"),
                               key=lambda p: p.stat().st_mtime) if model_output_dir.exists() else []
            if ckpt_dirs:
                resume_ckpt = str(ckpt_dirs[-1])
                print(f"  🔄 Resuming from checkpoint: {resume_ckpt}")

            # Use model-specific max_seq_length if set (e.g., 72B needs 1024)
            actual_max_length = getattr(spec, 'max_seq_length_override', 0) or max_length

            # Load tokenizer for this model to prepare data
            from transformers import AutoTokenizer
            model_path = spec.resolved_path
            logger.info(f"Loading tokenizer from: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                model_max_length=actual_max_length,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Tokenize for this model's tokenizer
            train_ds = prepare_tokenized_dataset(
                train_records, tokenizer, actual_max_length, template_name
            )
            eval_ds = prepare_tokenized_dataset(
                eval_records, tokenizer, actual_max_length, template_name
            )

            result = run_finetune(
                model_spec=spec,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                output_base_dir=output_base_dir,
                max_length=actual_max_length,
                template_name=template_name,
                use_unsloth=use_unsloth,
                report_to=report_to,
                resume_from_checkpoint=resume_ckpt,
                dry_run=dry_run,
            )
            results.append(result)
            result.print_summary()

        except Exception as e:
            logger.error(f"Failed: {spec.short_name} — {e}")
            results.append(TrainingResult(
                model_short_name=spec.short_name,
                model_id=spec.model_id,
                param_count=spec.param_count,
                status="failed",
                error_message=str(e),
            ))
        finally:
            # Free GPU memory between models to prevent OOM on next model
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared between experiment runs.")

    # Summary table
    print(f"\n\n{'═' * 70}")
    print(f"  EXPERIMENT SUITE SUMMARY")
    print(f"{'═' * 70}")
    print(f"{'Model':<25} {'Params':>8} {'Status':>12} {'Train Loss':>12} "
          f"{'Eval Loss':>12} {'Time (h)':>10}")
    print(f"{'─' * 79}")
    for r in results:
        hrs = r.training_time_seconds / 3600 if r.training_time_seconds else 0
        print(f"  {r.model_short_name:<23} {r.param_count:>8} {r.status:>12} "
              f"{r.train_loss:>12.4f} {r.eval_loss:>12.4f} {hrs:>10.2f}")

    return results
