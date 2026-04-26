"""
02-05 Instruction generator (main module).

Orchestrates full pipeline: load chunks, expert selection, domain knowledge, CoDE generation, quality filter, save.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import yaml

from .expert_selector_02_01 import ExpertSelector
from .domain_knowledge_02_02 import DomainKnowledgeInjector
from .coe_framework_02_03 import COEFramework, ExpertOutput
from .expert_agents_02_04 import create_expert, BaseExpert
from .meta_expert_02_09 import MetaExpert, TopicExtractor

logger = logging.getLogger(__name__)


class InstructionGenerator:
    """
    Instruction generator: load chunks -> expert selection -> domain knowledge -> CoDE -> quality filter -> save.
    """
    def __init__(
        self,
        config_path: Optional[str] = "configs/config.yaml",
        project_root: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            config_path: Config file path (used if config is None).
            project_root: Project root directory.
            config: Config dict (if provided, no file load).
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = Path(project_root)
        if config is not None:
            self.config = config
            self.config_path = None
        else:
            if config_path is None:
                config_path = "configs/config.yaml"
            self.config_path = self.project_root / config_path
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        self.coe_config = self.config.get("coe", {})
        self.data_config = self.config.get("data", {})
        self.expert_selector = ExpertSelector(
            use_ml_classifier=self.coe_config.get("use_adaptive_selection", True),
            classifier_model=self.coe_config.get("expert_selection_model", "bert-base-chinese"),
            min_experts=self.coe_config.get("min_experts", 1),
            max_experts=self.coe_config.get("max_experts", 3),
            use_layered_selection=self.coe_config.get("use_layered_selection", True),
            layered_config=self.coe_config.get("layered_selection", {})
        )
        knowledge_base_path = self.coe_config.get("knowledge_base_path") or self.data_config.get("knowledge_base_path")
        if knowledge_base_path and not Path(knowledge_base_path).is_absolute():
            knowledge_base_path = str(self.project_root / knowledge_base_path)
        
        self.knowledge_injector = DomainKnowledgeInjector(
            knowledge_base_path=knowledge_base_path,
            max_knowledge_items=self.coe_config.get("max_knowledge_items", 5)
        )
        self.experts = self._init_experts()
        self.coe_framework = COEFramework(
            experts=self.experts,
            enable_collaboration=self.coe_config.get("enable_collaboration", True),
            collaboration_mode=self.coe_config.get("collaboration_mode", "sequential"),
            max_iterations=self.coe_config.get("max_iterations", 2),
            quality_threshold=0.7
        )
        self.use_meta_expert = self.coe_config.get("use_meta_expert", True)
        if self.use_meta_expert:
            api_config = self.config.get("api", {})
            if not api_config:
                api_config = {
                    "provider": self.coe_config.get("api_provider", "google"),
                    "model_name": self.coe_config.get("model_name", "gemini-3-flash-preview"),
                    "api_key": None
                }
            self.topic_extractor = TopicExtractor(
                api_provider=api_config.get("provider", "google"),
                model_name=api_config.get("model_name", "gemini-3-flash-preview"),
                api_key=api_config.get("api_key")
            )
            self.meta_expert = MetaExpert(
                api_provider=api_config.get("provider", "google"),
                model_name=api_config.get("model_name", "gemini-3-flash-preview"),
                carbon_centric=self.coe_config.get("carbon_centric", True),
                api_key=api_config.get("api_key")
            )
            logger.info("Meta-Expert initialized")
        else:
            self.topic_extractor = None
            self.meta_expert = None
            logger.info("Meta-Expert disabled; using static templates")
    def _init_experts(self) -> Dict[str, BaseExpert]:
        """Initialize all expert agents."""
        experts = {}
        base_expert_names = [
            "qa_expert",
            "summary_expert",
            "extraction_expert",
            "classification_expert",
            "analysis_expert"
        ]
        # Phase 2 advanced experts (analysis + domain knowledge)
        phase2_advanced_experts = [
            "temporal_analysis_expert",
            "benchmark_comparison_expert",
            "greenwashing_detection_expert",
            "standard_alignment_expert",
            "knowledge_graph_expert",
            "consistency_verification_expert"
        ]
        # Phase 3 (tool-calling experts; not used in Phase 2)
        # "promise_performance_verification_expert"
        enable_advanced = self.coe_config.get("enable_advanced_experts", True)
        expert_names = base_expert_names + (phase2_advanced_experts if enable_advanced else [])
        
        for expert_name in expert_names:
            try:
                expert = create_expert(expert_name, self.coe_config)
                experts[expert_name] = expert
                logger.info(f"Initialized expert: {expert_name}")
            except Exception as e:
                logger.error(f"Failed to init expert {expert_name}: {e}")
        
        return experts
    
    def generate_for_chunk(
        self,
        chunk: Dict,
        use_knowledge: bool = True,
        use_adaptive_selection: bool = True
    ) -> Optional[Dict]:
        """Generate instruction-answer pair for one chunk. Returns None on failure."""
        chunk_text = chunk.get("text", "")
        if not chunk_text or len(chunk_text) < 50:
            logger.warning(f"Chunk too short, skip: {chunk.get('chunk_id', 'unknown')}")
            return None
        
        try:
            # 1. Extract core topics (Meta-Expert)
            core_topics = []
            section_name = None
            if self.use_meta_expert and self.topic_extractor:
                try:
                    core_topics = self.topic_extractor.extract_topics(chunk_text, max_topics=5)
                    logger.debug(f"Core topics: {core_topics}")
                    doc_id = chunk.get("doc_id", "")
                    if "治理" in doc_id:
                        section_name = "Governance Structure"
                    elif "员工" in doc_id or "社会" in doc_id:
                        section_name = "Social Responsibility"
                    elif "环境" in doc_id:
                        section_name = "Environmental Management"
                except Exception as e:
                    logger.warning(f"Topic extraction failed: {e}")
            # 2. Adaptive expert selection
            if use_adaptive_selection:
                selected_experts, selection_reasons = self.expert_selector.select_experts(chunk_text)
            else:
                selected_experts = list(self.experts.keys())
                selection_reasons = {"method": "all_experts"}
            if not selected_experts:
                logger.warning("No experts selected, skip")
                return None
            # 3. Meta-Expert dynamic instructions (if enabled)
            dynamic_instructions = {}
            if self.use_meta_expert and self.meta_expert:
                try:
                    for expert_name in selected_experts:
                        instruction = self.meta_expert.generate_instruction(
                            expert_type=expert_name,
                            chunk_text=chunk_text,
                            core_topics=core_topics,
                            section_name=section_name,
                            all_experts=selected_experts
                        )
                        dynamic_instructions[expert_name] = instruction
                        logger.debug(f"Meta-Expert instruction for {expert_name}: {instruction[:100]}...")
                except Exception as e:
                    logger.warning(f"Meta-Expert instruction generation failed: {e}")
            # 4. Retrieve domain knowledge
            context = {}
            if use_knowledge:
                knowledge_items = self.knowledge_injector.retrieve_knowledge(chunk_text)
                context["knowledge_items"] = knowledge_items
            # 5. Inject dynamic instructions into context
            if dynamic_instructions:
                context["dynamic_instructions"] = dynamic_instructions
                context["core_topics"] = core_topics
            # 6. CoDE framework generation
            output = self.coe_framework.generate(
                chunk_text=chunk_text,
                selected_experts=selected_experts,
                context=context
            )
            
            # 7. Build result
            result = {
                "instruction": output.instruction,
                "input": chunk_text,
                "output": output.response,
                "task_type": self._infer_task_type(selected_experts[0]),
                "source_chunk_id": chunk.get("chunk_id", ""),
                "source_doc_id": chunk.get("doc_id", ""),
                "metadata": {
                    "generation_method": "coe_meta_expert" if self.use_meta_expert else "coe",
                    "experts": selected_experts,
                    "selection_reasons": selection_reasons,
                    "quality_score": output.quality_score,
                    "knowledge_used": use_knowledge,
                    "meta_expert_used": self.use_meta_expert,
                    "core_topics": core_topics,
                    "dynamic_instructions": dynamic_instructions if dynamic_instructions else None,
                    "expert_count": len(selected_experts),
                    **output.metadata
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Instruction generation failed (chunk_id: {chunk.get('chunk_id', 'unknown')}): {e}")
            return None
    def _infer_task_type(self, expert_name: str) -> str:
        """Infer task type from expert name."""
        mapping = {
            "qa_expert": "qa",
            "summary_expert": "summarization",
            "extraction_expert": "information_extraction",
            "classification_expert": "classification",
            "analysis_expert": "analysis"
        }
        return mapping.get(expert_name, "unknown")
    
    def generate_batch(
        self,
        chunks: List[Dict],
        output_file: str,
        sample_size: Optional[int] = None,
        checkpoint_file: Optional[str] = None
    ) -> Dict:
        """Batch generate instruction-answer pairs. Returns stats dict."""
        if sample_size and sample_size > 0:
            import random
            if len(chunks) > sample_size:
                chunks = random.sample(chunks, sample_size)
        processed_chunk_ids = set()
        if checkpoint_file and Path(checkpoint_file).exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                processed_chunk_ids = set(checkpoint.get("processed_chunk_ids", []))
            logger.info(f"Resumed from checkpoint; {len(processed_chunk_ids)} chunks already processed")
        
        stats = {
            "total_chunks": len(chunks),
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = "a" if processed_chunk_ids else "w"
        with open(output_path, mode, encoding="utf-8") as f_out:
            for chunk in tqdm(chunks, desc="Generating instructions"):
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id in processed_chunk_ids:
                    stats["skipped"] += 1
                    continue
                
                stats["processed"] += 1
                
                try:
                    result = self.generate_for_chunk(chunk)
                except ValueError as e:
                    if "每日请求限制" in str(e) or "daily" in str(e).lower():
                        logger.error(str(e))
                        logger.info("Suggestions: wait until tomorrow, or set up paid account for higher quota.")
                        logger.info(f"Processed {stats['successful']} chunks saved to checkpoint.")
                        break
                    else:
                        raise
                
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stats["successful"] += 1
                    processed_chunk_ids.add(chunk_id)
                else:
                    stats["failed"] += 1
                if checkpoint_file and stats["processed"] % 100 == 0:
                    self._save_checkpoint(checkpoint_file, processed_chunk_ids, stats)
        if checkpoint_file:
            self._save_checkpoint(checkpoint_file, processed_chunk_ids, stats)
            if Path(checkpoint_file).exists():
                Path(checkpoint_file).unlink()
                logger.info("Done; checkpoint file removed")
        
        return stats
    
    def _save_checkpoint(self, checkpoint_file: str, processed_chunk_ids: set, stats: Dict):
        """Save checkpoint."""
        checkpoint = {
            "processed_chunk_ids": list(processed_chunk_ids),
            "stats": stats
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def _extract_layer(self, doc_id: str) -> str:
        """Extract layer from doc_id (supports both English and legacy Chinese names)."""
        if not doc_id:
            return "unknown"
        parts = doc_id.split("_", 1)
        if not parts:
            return "unknown"
        first = parts[0]
        if first in ["Layer1", "Layer2", "Layer3", "Layer4"]:
            return first
        legacy = {"第一层": "Layer1", "第二层": "Layer2", "第三层": "Layer3", "第四层": "Layer4"}
        return legacy.get(first, "unknown")
    def _extract_year(self, doc_id: str) -> int:
        """Extract year from doc_id."""
        if not doc_id:
            return 0
        import re
        pattern1 = r'_(\d{4})_'
        match = re.search(pattern1, doc_id)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2100:
                return year
        pattern2 = r'(\d{4})年'
        match = re.search(pattern2, doc_id)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2100:
                return year
        return 0
    
    def _filter_chunks_by_layer(
        self,
        chunks: List[Dict],
        layer_config: Dict[str, Any]
    ) -> List[Dict]:
        """Filter chunks by layer config."""
        import random
        layer_chunks = {}
        layer_stats = {}
        
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "")
            layer = self._extract_layer(doc_id)
            
            if layer not in layer_stats:
                layer_stats[layer] = {"total": 0, "included": 0}
            layer_stats[layer]["total"] += 1
            
            if layer not in layer_chunks:
                layer_chunks[layer] = []
            layer_chunks[layer].append(chunk)
        
        filtered_chunks = []
        for layer, layer_chunk_list in layer_chunks.items():
            if layer not in layer_config:
                continue
            
            config = layer_config[layer]
            if not config.get("include", False):
                continue
            
            sample_ratio = config.get("sample_ratio", 1.0)
            
            if layer == "Layer2" and config.get("select_by_year", False):
                year_weights = config.get("year_weights", {
                    "2024": 0.40,
                    "2023": 0.30,
                    "2022": 0.20,
                    "2021": 0.10
                })
                target_total = config.get("target_total", 50000)
                year_range = config.get("year_range", [2021, 2024])
                chunks_by_year = {}
                for chunk in layer_chunk_list:
                    doc_id = chunk.get("doc_id", "")
                    year = self._extract_year(doc_id)
                    if year_range[0] <= year <= year_range[1]:
                        if year not in chunks_by_year:
                            chunks_by_year[year] = []
                        chunks_by_year[year].append(chunk)
                
                selected_chunks = []
                year_dist = {}
                for year_str, weight in sorted(year_weights.items(), key=lambda x: int(x[0]), reverse=True):
                    year = int(year_str)
                    if year not in chunks_by_year:
                        continue
                    
                    target_count = int(target_total * weight)
                    available_chunks = chunks_by_year[year]
                    
                    available_chunks.sort(key=lambda c: self._extract_year(c.get("doc_id", "")), reverse=True)
                    if len(available_chunks) <= target_count:
                        selected = available_chunks
                    else:
                        selected = random.sample(available_chunks, target_count)
                    
                    selected_chunks.extend(selected)
                    year_dist[year] = len(selected)
                
                logger.info(f"  Layer 2 year-weighted: target {target_total:,} chunks, range {year_range[0]}-{year_range[1]}")
                logger.info(f"    Selected: {len(selected_chunks):,} chunks; year dist: {dict(sorted(year_dist.items(), reverse=True))}")
                
                filtered_chunks.extend(selected_chunks)
                layer_stats[layer]["included"] = len(selected_chunks)
            else:
                if sample_ratio >= 1.0:
                    filtered_chunks.extend(layer_chunk_list)
                    layer_stats[layer]["included"] = len(layer_chunk_list)
                else:
                    target_count = int(len(layer_chunk_list) * sample_ratio)
                    selected = random.sample(layer_chunk_list, min(target_count, len(layer_chunk_list)))
                    filtered_chunks.extend(selected)
                    layer_stats[layer]["included"] = len(selected)
        
        logger.info("="*60)
        logger.info("Layer filter stats:")
        logger.info("="*60)
        for layer, stats in sorted(layer_stats.items()):
            total = stats["total"]
            included = stats["included"]
            ratio = included / total * 100 if total > 0 else 0
            logger.info(f"  {layer}: {included:,}/{total:,} ({ratio:.1f}%)")
        logger.info(f"  Total: {len(filtered_chunks):,}/{len(chunks):,} chunks")
        logger.info("="*60)
        
        return filtered_chunks
    
    def generate_from_file(
        self,
        chunks_file: str,
        output_file: str,
        sample_size: Optional[int] = None,
        checkpoint_file: Optional[str] = None,
        layer_config: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Read chunks from file and run batch generation. Returns stats."""
        chunks_file_path = Path(chunks_file)
        if not chunks_file_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        all_chunks = []
        logger.info(f"Reading chunks: {chunks_file}")
        with open(chunks_file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading chunks"):
                try:
                    chunk = json.loads(line)
                    all_chunks.append(chunk)
                except Exception:
                    continue
        
        logger.info(f"Read {len(all_chunks)} chunks.")
        if layer_config:
            chunks_to_process = self._filter_chunks_by_layer(all_chunks, layer_config)
            logger.info(f"After layer filter: {len(chunks_to_process)} chunks to process.")
        else:
            chunks_to_process = all_chunks
        return self.generate_batch(chunks_to_process, output_file, sample_size, checkpoint_file)

