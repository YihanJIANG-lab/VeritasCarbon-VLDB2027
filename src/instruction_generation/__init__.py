"""
Instruction generation module.

Includes CoDE framework, expert agents, evaluation metrics, and related utilities.
"""

from .instruction_generator_02_05 import InstructionGenerator
from .expert_selector_02_01 import ExpertSelector
from .domain_knowledge_02_02 import DomainKnowledgeInjector
from .coe_framework_02_03 import COEFramework, ExpertOutput
from .expert_agents_02_04 import (
    BaseExpert, QAExpert, SummaryExpert, ExtractionExpert,
    ClassificationExpert, AnalysisExpert, create_expert
)
from .evaluation_metrics_02_06 import MultiMetricEvaluator
from .baseline_experiments_02_07 import BaselineExperiment
from .ablation_studies_02_08 import AblationStudy

__all__ = [
    "InstructionGenerator",
    "ExpertSelector",
    "DomainKnowledgeInjector",
    "COEFramework",
    "ExpertOutput",
    "BaseExpert",
    "QAExpert",
    "SummaryExpert",
    "ExtractionExpert",
    "ClassificationExpert",
    "AnalysisExpert",
    "create_expert",
    "MultiMetricEvaluator",
    "BaselineExperiment",
    "AblationStudy"
]