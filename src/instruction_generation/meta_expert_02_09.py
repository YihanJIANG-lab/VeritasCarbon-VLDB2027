"""
02-09 Meta-Expert: dynamic instruction generation.

Two-stage generation for diverse, context-aware expert instructions.
"""

import logging
import os
from typing import Dict, Optional, Any, List
import re

logger = logging.getLogger(__name__)


class TopicExtractor:
    """Extract core entities and topics from chunk text."""
    def __init__(self, api_provider: str = "google", model_name: str = "gemini-3-flash-preview", **kwargs):
        """
        Args:
            api_provider: API provider
            model_name: Model name
            **kwargs: e.g. api_key
        """
        self.api_provider = api_provider
        self.model_name = model_name
        self.api_client = None
        self._init_api_client(**kwargs)
    
    def _init_api_client(self, **kwargs):
        """Initialize API client."""
        if self.api_provider == "google":
            try:
                import google.genai as genai
                api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    self.api_client = genai.Client(api_key=api_key)
                    logger.info("TopicExtractor API client initialized")
                else:
                    logger.error("GOOGLE_API_KEY not set")
            except ImportError:
                logger.error("google-genai not installed")
    def extract_topics(self, chunk_text: str, max_topics: int = 5) -> List[str]:
        """Extract core topics (keywords) from chunk text."""
        if not self.api_client:
            return self._extract_topics_simple(chunk_text, max_topics)
        try:
            prompt = f"""You are an ESG (environment, social, governance) topic extraction expert.
Task: Extract core themes from the text below in at most {max_topics} keywords.
Requirements:
1. Keywords should reflect the main issues in the text.
2. Prefer ESG-related themes.
3. Output keywords separated by commas only, no other content.

Text:
{chunk_text[:1000]}

Keywords:"""
            
            response = self.api_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 100
                }
            )
            
            topics_text = response.text.strip()
            topics = [t.strip() for t in topics_text.split(",") if t.strip()][:max_topics]
            return topics if topics else self._extract_topics_simple(chunk_text, max_topics)
        except Exception as e:
            logger.warning(f"Topic extraction API failed, using simple method: {e}")
            return self._extract_topics_simple(chunk_text, max_topics)
    def _extract_topics_simple(self, chunk_text: str, max_topics: int) -> List[str]:
        """Simple keyword extraction (fallback)."""
        esg_keywords = [
            "碳排放", "碳中和", "碳达峰", "环境", "社会", "治理", "ESG", "CSR",
            "可持续发展", "环保", "社会责任", "公司治理", "员工", "培训",
            "供应链", "合规", "风险", "创新", "质量", "安全", "能源", "水资源"
        ]
        
        found_keywords = []
        text_lower = chunk_text.lower()
        for kw in esg_keywords:
            if kw in text_lower and kw not in found_keywords:
                found_keywords.append(kw)
                if len(found_keywords) >= max_topics:
                    break
        
        return found_keywords if found_keywords else ["ESG", "sustainability"]


class MetaExpert:
    """Meta-Expert: dynamic instruction generator."""
    def __init__(
        self,
        api_provider: str = "google",
        model_name: str = "gemini-3-flash-preview",
        carbon_centric: bool = True,
        **kwargs
    ):
        """
        Args:
            api_provider: API provider
            model_name: Model name
            carbon_centric: Enable carbon-centric constraint
            **kwargs: e.g. api_key
        """
        self.api_provider = api_provider
        self.model_name = model_name
        self.carbon_centric = carbon_centric
        self.api_client = None
        self._init_api_client(**kwargs)
    
    def _init_api_client(self, **kwargs):
        """Initialize API client."""
        if self.api_provider == "google":
            try:
                import google.genai as genai
                import os
                api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    self.api_client = genai.Client(api_key=api_key)
                    logger.info("MetaExpert API client initialized")
                else:
                    logger.error("GOOGLE_API_KEY not set")
            except ImportError:
                logger.error("google-genai not installed")
    def generate_instruction(
        self,
        expert_type: str,
        chunk_text: str,
        core_topics: List[str],
        section_name: Optional[str] = None,
        all_experts: Optional[List[str]] = None
    ) -> str:
        """Generate dynamic instruction for the given expert. Returns default if API unavailable."""
        if not self.api_client:
            logger.warning("MetaExpert API not initialized; using default instruction")
            return self._get_default_instruction(expert_type)
        topics_str = ", ".join(core_topics) if core_topics else "unspecified"
        section_info = f"from report section [{section_name}]" if section_name else "from the report"
        collaboration_info = ""
        if all_experts and len(all_experts) > 1:
            other_experts = [e for e in all_experts if e != expert_type]
            collaboration_info = f"""
- **Collaboration**: This run uses multiple experts: {', '.join(all_experts)}
- **Current expert**: {expert_type}
- **Others**: {', '.join(other_experts) if other_experts else 'none'}
- **Requirement**: Your instruction should match {expert_type}'s role and complement others (e.g. extraction vs analysis; avoid duplication).
"""
        carbon_constraint = ""
        if self.carbon_centric:
            carbon_constraint = """
1. **Carbon-centric**: The instruction must lead to assessing the company's carbon performance or credibility of carbon commitments.
2. **Cross-domain**: Use information in this chunk (even if not carbon-related) as evidence for carbon strategy.
3. **Depth**: Guide the analyst to compare, evaluate, predict, or critique—not just extract.
"""
        else:
            carbon_constraint = """
1. **Relevant**: Instruction must match the chunk's main themes.
2. **Depth**: Guide comparison, evaluation, prediction, or critique.
3. **Clear**: One clear question or task.
"""
        expert_descriptions = {
            "qa_expert": "QA generation expert",
            "summary_expert": "Summarization expert",
            "extraction_expert": "Information extraction expert for ESG metrics",
            "classification_expert": "ESG dimension classification expert",
            "analysis_expert": "Basic analysis expert",
            "temporal_analysis_expert": "Temporal and trend analysis expert",
            "benchmark_comparison_expert": "Benchmark comparison expert",
            "greenwashing_detection_expert": "Greenwashing detection expert",
            "standard_alignment_expert": "ESG standard alignment expert",
            "knowledge_graph_expert": "Knowledge graph / entity-relation expert",
            "consistency_verification_expert": "Consistency verification expert",
            "promise_performance_verification_expert": "Promise-vs-performance verification and tool-call expert"
        }
        expert_desc = expert_descriptions.get(expert_type, "analyst")
        prompt = f"""# Role
You are a top carbon-strategy analyst. Your task is to evaluate all information related to the company's carbon emissions and carbon-reduction commitments. Now design one work instruction for a subordinate analyst.

# Context
- Subordinate role: {expert_type} ({expert_desc})
- Chunk is {section_info}
- Core topics: {topics_str}
- Preview: {chunk_text[:500]}...
{collaboration_info}

# Task
Generate one work instruction for this analyst. The instruction must:
{carbon_constraint}

# Output
Work instruction:"""
        
        try:
            response = self.api_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 200
                }
            )
            instruction = response.text.strip()
            instruction = re.sub(r'^工作指令[:：]\s*', '', instruction)
            instruction = re.sub(r'^Work instruction[:：]\s*', '', instruction, flags=re.I)
            return instruction if instruction else self._get_default_instruction(expert_type)
        except Exception as e:
            logger.error(f"MetaExpert instruction generation failed: {e}")
            return self._get_default_instruction(expert_type)
    def _get_default_instruction(self, expert_type: str) -> str:
        """Default instructions (fallback)."""
        defaults = {
            "qa_expert": "Generate a high-quality Q&A pair based on the text.",
            "summary_expert": "Summarize the main content of the text.",
            "extraction_expert": "Extract ESG-related metrics from the text.",
            "classification_expert": "Classify the text by ESG dimension (environment/social/governance).",
            "analysis_expert": "Analyze ESG-related content in the text.",
            "temporal_analysis_expert": "Analyze temporal patterns, commitment trends, and data continuity.",
            "benchmark_comparison_expert": "Compare carbon commitments in the text with industry benchmarks.",
            "greenwashing_detection_expert": "Identify greenwashing patterns (e.g. vague commitments, selective disclosure).",
            "standard_alignment_expert": "Identify ESG standards followed and assess compliance.",
            "knowledge_graph_expert": "Extract ESG entities and relations for a knowledge graph.",
            "consistency_verification_expert": "Verify data consistency and identify contradictions or gaps.",
            "promise_performance_verification_expert": "Identify verifiable claims and generate tool-call instructions for verification."
        }
        return defaults.get(expert_type, "Analyze the following text.")

