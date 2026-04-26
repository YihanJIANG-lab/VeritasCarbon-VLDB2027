"""
02-04 Expert agents.

Implements 5 experts: QA, Summary, Extraction, Classification, Analysis.
"""

import logging
import os
from typing import Dict, Optional, Any
import json
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """API rate limiter for free-tier quotas."""
    def __init__(self):
        self.last_request_time = 0
        self.request_times = []
        self.daily_requests = []
        self.min_delay = 12
        self.max_rpm = 5
        self.max_rpd = 20

    def wait_if_needed(self):
        """Wait if needed to stay within rate limits."""
        now = time.time()
        today = datetime.now().date()
        self.request_times = [t for t in self.request_times if now - t < 60]
        self.daily_requests = [dt for dt in self.daily_requests if dt.date() == today]
        if len(self.daily_requests) >= self.max_rpd:
            tomorrow = (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0)
            wait_seconds = (tomorrow - datetime.now()).total_seconds()
            logger.warning(f"Daily request limit ({self.max_rpd}) reached; wait {wait_seconds/3600:.1f}h")
            raise ValueError(f"Daily limit ({self.max_rpd}) reached; try tomorrow or use paid account")
        if len(self.request_times) >= self.max_rpm:
            oldest_request = min(self.request_times)
            wait_time = 60 - (now - oldest_request) + 1
            logger.info(f"RPM limit reached, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            logger.debug(f"Waiting {wait_time:.1f}s for rate limit...")
            time.sleep(wait_time)
        self.last_request_time = time.time()
        self.request_times.append(self.last_request_time)
        self.daily_requests.append(datetime.now())


_rate_limiter = RateLimiter()
from .coe_framework_02_03 import ExpertOutput


class BaseExpert:
    """Base class for expert agents."""

    def __init__(
        self,
        name: str,
        api_provider: str = "openai",
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Args:
            name: Expert name
            api_provider: API provider (openai, anthropic, zhipu, google)
            model_name: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens
        """
        self.name = name
        self.api_provider = api_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_client = None
        self._init_api_client()
    
    def _init_api_client(self):
        """Initialize API client."""
        if self.api_provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error("OPENAI_API_KEY not set; export OPENAI_API_KEY='your-key'")
                else:
                    self.api_client = openai.OpenAI(api_key=api_key)
                    logger.info(f"OpenAI API initialized (model: {self.model_name})")
            except ImportError:
                logger.error("openai not installed; pip install openai")
        elif self.api_provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.error("ANTHROPIC_API_KEY not set")
                else:
                    self.api_client = anthropic.Anthropic(api_key=api_key)
                    logger.info(f"Anthropic API initialized (model: {self.model_name})")
            except ImportError:
                logger.error("anthropic not installed; pip install anthropic")
        elif self.api_provider == "zhipu":
            try:
                from zhipuai import ZhipuAI
                api_key = os.getenv("ZHIPU_API_KEY")
                if not api_key:
                    logger.error("ZHIPU_API_KEY not set")
                else:
                    self.api_client = ZhipuAI(api_key=api_key)
                    logger.info(f"Zhipu API initialized (model: {self.model_name})")
            except ImportError:
                logger.error("zhipuai not installed; pip install zhipuai")
        elif self.api_provider == "google":
            try:
                from google import genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    logger.error("GOOGLE_API_KEY not set")
                else:
                    self.api_client = genai.Client(api_key=api_key)
                    logger.info(f"Google Gemini API initialized (model: {self.model_name})")
            except ImportError:
                logger.error("google-genai not installed; pip install google-genai")
        else:
            logger.warning(f"Unknown API provider: {self.api_provider}")
    
    def _call_api(self, prompt: str, max_retries: int = 3) -> str:
        """Call API to generate text; retries and rate limiting."""
        if not self.api_client:
            error_msg = f"API client not initialized: {self.api_provider}"
            logger.error(error_msg)
            logger.error("Check OPENAI_API_KEY, ANTHROPIC_API_KEY, ZHIPU_API_KEY, or GOOGLE_API_KEY")
            raise ValueError(error_msg)
        if self.api_provider == "google":
            try:
                _rate_limiter.wait_if_needed()
            except ValueError as e:
                logger.error(str(e))
                raise
        try:
            import openai
        except ImportError:
            openai = None
        
        for attempt in range(max_retries):
            try:
                if self.api_provider == "openai":
                    response = self.api_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    content = response.choices[0].message.content
                    if not content:
                        logger.warning("OpenAI API returned empty content")
                    return content or ""
                elif self.api_provider == "anthropic":
                    response = self.api_client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    if not content:
                        logger.warning("Anthropic API returned empty content")
                    return content or ""
                elif self.api_provider == "zhipu":
                    response = self.api_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    content = response.choices[0].message.content
                    if not content:
                        logger.warning("Zhipu API returned empty content")
                    return content or ""
                elif self.api_provider == "google":
                    response = self.api_client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens
                        }
                    )
                    content = response.text
                    if not content:
                        logger.warning("Google Gemini API returned empty content")
                    return content or ""
                else:
                    raise ValueError(f"Unsupported API provider: {self.api_provider}")
                    
            except Exception as e:
                error_str = str(e)
                
                # RateLimitError (429)
                is_rate_limit = False
                if openai:
                    try:
                        if isinstance(e, openai.RateLimitError):
                            is_rate_limit = True
                    except:
                        pass
                
                # Fallback: check error message
                if not is_rate_limit and ("429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower()):
                    is_rate_limit = True
                
                # SSL/network error
                is_network_error = False
                if any(keyword in error_str for keyword in ["SSL", "EOF", "connection", "timeout", "network", "Unexpected EOF"]):
                    is_network_error = True
                
                if is_rate_limit:
                    wait_time = (2 ** attempt) * 10
                    if attempt < max_retries - 1:
                        logger.warning(f"API rate limit (429), waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    logger.error(f"API call failed ({self.api_provider}): RateLimitError (429). Check billing, rate limits.")
                    raise
                elif is_network_error:
                    wait_time = (2 ** attempt) * 5
                    if attempt < max_retries - 1:
                        logger.warning(f"Network/SSL error ({type(e).__name__}), waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        logger.debug(f"Details: {e}")
                        time.sleep(wait_time)
                        continue
                    logger.error(f"API call failed ({self.api_provider}): network/SSL, max retries. Check network, firewall, proxy.")
                    raise
                else:
                    logger.error(f"API call failed ({self.api_provider}): {e}. Check API key, network, quota.")
                    raise
        raise RuntimeError("API call failed: max retries exceeded")
    
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Get prompt template (override in subclass)."""
        raise NotImplementedError

    def parse_response(self, response: str) -> ExpertOutput:
        """Parse API response (override in subclass)."""
        raise NotImplementedError

    def generate(self, chunk_text: str, context: Optional[Dict] = None) -> ExpertOutput:
        """Generate instruction-answer pair. Args: chunk_text, context. Returns: ExpertOutput."""
        try:
            dynamic_instruction = None
            if context and context.get("dynamic_instruction"):
                dynamic_instruction = context["dynamic_instruction"]
                logger.debug(f"Expert {self.name} using Meta-Expert dynamic instruction")
            prompt = self.get_prompt_template(chunk_text, context, dynamic_instruction=dynamic_instruction)
            logger.debug(f"Expert {self.name} calling API...")
            response = self._call_api(prompt)
            if not response or len(response.strip()) == 0:
                logger.warning(f"Expert {self.name} API returned empty response")
                return ExpertOutput(
                    expert_name=self.name,
                    instruction="",
                    response="",
                    quality_score=0.0,
                    metadata={"expert": self.name, "error": "empty_response", "raw_response": ""}
                )
            
            logger.debug(f"Expert {self.name} API success, response length: {len(response)}")
            result = self.parse_response(response)
            if not result.instruction or not result.response:
                logger.warning(f"Expert {self.name} parse incomplete: instruction={bool(result.instruction)}, response={bool(result.response)}")
                logger.debug(f"Raw response: {response[:200]}...")
            return result
        except Exception as e:
            logger.error(f"Expert {self.name} generation failed: {e}", exc_info=True)
            return ExpertOutput(
                expert_name=self.name,
                instruction="",
                response="",
                quality_score=0.0,
                metadata={"expert": self.name, "error": str(e), "raw_response": ""}
            )


class QAExpert(BaseExpert):
    """QA expert: generates Q&A pairs."""
    
    def __init__(self, **kwargs):
        super().__init__(name="qa_expert", **kwargs)
    
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build QA generation prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction below and the text, generate one high-quality Q&A pair.
Work instruction: {dynamic_instruction}
Requirements:
1. The question must align with the work instruction.
2. The answer must be grounded in the text or reasonable inference.
3. The question should be specific and encourage deep thinking.
4. The answer should be accurate and complete, with cross-domain analysis."""
        else:
            task_description = """Task: Generate one high-quality Q&A pair from the given text.
Requirements:
1. The question should be highly relevant to the text.
2. The answer must be findable in the text or reasonably inferred.
3. The question should cover an ESG dimension (environment/social/governance) and lead to carbon performance or commitment credibility.
4. The question should be specific and clear.
5. The answer should be accurate and complete."""
        prompt = f"""You are a senior ESG (environment, social, governance) Q&A generation expert.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Question: [your question]
Answer: [your answer]
Output only the Q&A pair."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse QA response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        question = ""
        answer = ""
        for line in lines:
            if line.startswith("问题：") or line.startswith("问题:") or line.strip().lower().startswith("question:"):
                question = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("答案：") or line.startswith("答案:") or line.strip().lower().startswith("answer:"):
                answer = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not question or not answer:
            parts = response.split("\n\n")
            if len(parts) >= 2:
                question = parts[0].replace("问题：", "").replace("问题:", "").strip()
                if ":" in question:
                    question = question.split(":", 1)[-1].strip()
                answer = parts[1].replace("答案：", "").replace("答案:", "").strip()
                if ":" in answer:
                    answer = answer.split(":", 1)[-1].strip()
        return ExpertOutput(
            instruction=question if question else "Answer the following question.",
            response=answer if answer else response,
            metadata={"expert": "qa_expert", "raw_response": response}
        )


class SummaryExpert(BaseExpert):
    """Summary expert: generates summarization instruction-answer pairs."""
    def __init__(self, **kwargs):
        super().__init__(name="summary_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build Summary prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate one summarization task (instruction-answer pair).
Work instruction: {dynamic_instruction}
Requirements:
1. The instruction should align with the work instruction.
2. The answer should be a 100-200 word summary highlighting content relevant to the instruction.
3. The summary should cover core information and cross-domain analysis.
4. Highlight ESG content and lead to carbon performance or commitment credibility."""
        else:
            task_description = """Task: Generate one summarization task (instruction-answer pair) from the text.
Requirements:
1. The instruction should be like "Summarize the main content of the following text" and lead to carbon/ESG assessment.
2. The answer should be a 100-200 word summary.
3. Cover core information and highlight ESG content."""
        prompt = f"""You are a senior ESG summarization expert.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [your instruction]
Summary: [your summary]
Output only instruction and summary."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse Summary response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Summarize the main content of the following text."
        summary = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("摘要：") or line.startswith("摘要:") or line.strip().lower().startswith("summary:"):
                summary = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not summary:
            summary = response.strip()
        return ExpertOutput(
            expert_name="summary_expert",
            instruction=instruction,
            response=summary,
            quality_score=0.0,
            metadata={"expert": "summary_expert", "raw_response": response}
        )


class ExtractionExpert(BaseExpert):
    """Extraction expert: information extraction instruction-answer pairs."""
    def __init__(self, **kwargs):
        super().__init__(name="extraction_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build Extraction prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate one information-extraction task (instruction-answer pair).
Work instruction: {dynamic_instruction}
Requirements:
1. The instruction should align with the work instruction; focus on carbon performance or commitment-related information.
2. The answer should contain structured extracted information.
3. Information should be accurate and complete for cross-domain analysis.
4. If no relevant information exists, answer "No relevant information found" and briefly explain."""
        else:
            task_description = """Task: Generate one information-extraction task (instruction-answer pair) from the text.
Requirements:
1. The instruction should ask for specific ESG information (e.g. carbon emissions, employee count, training hours) and lead to carbon/commitment assessment.
2. The answer should contain structured extracted information.
3. Information should be accurate and complete.
4. If no relevant information exists, answer "No relevant information found"."""
        prompt = f"""You are a senior ESG information extraction expert.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [your instruction]
Extraction result: [structured information]
Output only instruction and extraction result."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse Extraction response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Extract ESG-related metrics from the following text."
        extraction = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif "提取结果" in line or line.strip().lower().startswith("extraction result:"):
                extraction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not extraction:
            extraction = response.strip()
        return ExpertOutput(
            expert_name="extraction_expert",
            instruction=instruction,
            response=extraction,
            quality_score=0.0,
            metadata={"expert": "extraction_expert", "raw_response": response}
        )


class ClassificationExpert(BaseExpert):
    """Classification expert: ESG dimension classification (E/S/G) instruction-answer pairs."""
    def __init__(self, **kwargs):
        super().__init__(name="classification_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build Classification prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate one classification task (instruction-answer pair).
Work instruction: {dynamic_instruction}
Requirements:
1. The instruction should align with the work instruction and relate to carbon performance or commitments.
2. The answer should classify the text by ESG dimension (environment/social/governance) and explain how it affects carbon/commitment credibility.
3. Classification should be accurate with clear justification.
4. If multiple dimensions apply, explain each dimension's relation to carbon."""
        else:
            task_description = """Task: Generate one classification task (instruction-answer pair) from the text.
Requirements:
1. The instruction should ask to classify the text by ESG dimension (environment/social/governance) and lead to carbon/commitment assessment.
2. The answer should state the dimension(s) and briefly justify.
3. If multiple dimensions apply, answer "Mixed" and explain relations to carbon."""
        prompt = f"""You are a senior ESG classification expert.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [your instruction]
Classification result: [Environment/Social/Governance/Mixed] + rationale
Output only instruction and classification result."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse Classification response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Classify the following text by ESG dimension (environment/social/governance)."
        classification = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif "分类结果" in line or "classification result" in line.lower():
                classification = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not classification:
            classification = response.strip()
        return ExpertOutput(
            expert_name="classification_expert",
            instruction=instruction,
            response=classification,
            quality_score=0.0,
            metadata={"expert": "classification_expert", "raw_response": response}
        )


class AnalysisExpert(BaseExpert):
    """Analysis expert: comparative/critical analysis instruction-answer pairs."""
    def __init__(self, **kwargs):
        super().__init__(name="analysis_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build Analysis prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate one analysis task (instruction-answer pair).
Work instruction: {dynamic_instruction}
Requirements:
1. The instruction should align with the work instruction and focus on carbon performance or commitments.
2. The answer should contain in-depth analysis and cross-domain reasoning.
3. Cover multiple ESG dimensions and how they affect carbon performance.
4. Be critical and constructive; assess credibility of carbon commitments."""
        else:
            task_description = """Task: Generate one analysis task (instruction-answer pair) from the text.
Requirements:
1. The instruction should ask to analyze ESG content (e.g. comparison, trend, impact) and lead to carbon/commitment assessment.
2. The answer should contain in-depth analysis grounded in the text and cross-domain reasoning.
3. Be critical and assess credibility of carbon commitments."""
        prompt = f"""You are a senior ESG analysis expert.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [your instruction]
Analysis result: [your analysis]
Output only instruction and analysis result."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse Analysis response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Analyze the ESG-related content in the following text."
        analysis = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif "分析结果" in line or "analysis result" in line.lower():
                analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not analysis:
            analysis = response.strip()
        return ExpertOutput(
            expert_name="analysis_expert",
            instruction=instruction,
            response=analysis,
            quality_score=0.0,
            metadata={"expert": "analysis_expert", "raw_response": response}
        )


class TemporalAnalysisExpert(BaseExpert):
    """Temporal analysis expert: multi-year comparison and trend analysis."""
    def __init__(self, **kwargs):
        super().__init__(name="temporal_analysis_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build temporal analysis prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate a temporal analysis task.
Work instruction: {dynamic_instruction}
Requirements:
1. Identify time information (years, periods) in the text.
2. Extract temporal features of commitments, data, and actions.
3. Generate a cross-year comparison or trend analysis task."""
        else:
            task_description = """Task: Generate a temporal analysis task (instruction-answer pair) from the text.
Requirements:
1. Identify time information (years, periods).
2. Extract temporal features of commitments, data, and actions.
3. Analyze commitment trend (weakening vs strengthening).
4. Check data continuity (gaps).
5. Lead to assessment of carbon commitment credibility."""
        prompt = f"""You are a senior ESG temporal analysis expert, skilled at multi-year commitment and data trends.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [analysis instruction: compare years or identify trends]
Answer: [analysis: temporal features, trend judgment, consistency assessment]
Output only instruction and answer."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse temporal analysis response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Analyze temporal features and cross-year trends in the following text."
        analysis = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("回答：") or line.startswith("回答:") or line.strip().lower().startswith("answer:"):
                analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not analysis:
            analysis = response.strip()
        return ExpertOutput(
            expert_name="temporal_analysis_expert",
            instruction=instruction,
            response=analysis,
            quality_score=0.0,
            metadata={"expert": "temporal_analysis_expert", "raw_response": response}
        )


class BenchmarkComparisonExpert(BaseExpert):
    """Benchmark comparison expert: industry-level comparison analysis."""
    def __init__(self, **kwargs):
        super().__init__(name="benchmark_comparison_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build benchmark comparison prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate an industry benchmark comparison task.
Work instruction: {dynamic_instruction}
Requirements:
1. Identify the company's industry.
2. Extract company commitments and data.
3. Generate industry benchmark comparison task."""
        else:
            task_description = """Task: Generate an industry benchmark comparison task (instruction-answer pair) from the text.
Requirements:
1. Identify company industry if present in the text.
2. Extract carbon commitments and emission data.
3. Compare with industry average and assess reasonableness.
4. Flag anomalous commitments (too high or too low).
5. Lead to carbon commitment credibility assessment."""
        prompt = f"""You are a senior ESG benchmark comparison expert, familiar with industry ESG standards and averages.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [comparison instruction: compare with industry average]
Answer: [comparison result: industry, benchmark comparison, anomaly judgment]
Output only instruction and answer."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse benchmark comparison response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Compare the carbon commitments in the text with industry average."
        analysis = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("回答：") or line.startswith("回答:") or line.strip().lower().startswith("answer:"):
                analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not analysis:
            analysis = response.strip()
        return ExpertOutput(
            expert_name="benchmark_comparison_expert",
            instruction=instruction,
            response=analysis,
            quality_score=0.0,
            metadata={"expert": "benchmark_comparison_expert", "raw_response": response}
        )


class GreenwashingDetectionExpert(BaseExpert):
    """Greenwashing detection expert: identify greenwashing patterns."""
    def __init__(self, **kwargs):
        super().__init__(name="greenwashing_detection_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build greenwashing detection prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        greenwashing_patterns = """
Common greenwashing patterns:
1. Vague commitments: "committed to", "plan to" without concrete targets
2. Selective disclosure: only favorable data, hiding unfavorable information
3. Exaggerated impact: overstating environmental project effects
4. Time mismatch: presenting future plans as achieved results
5. Scope confusion: mixing scope 1/2/3 emissions
6. Inconsistency: commitments vs actual actions
7. Lack of verification: no third-party or traceable evidence"""
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate a greenwashing detection task.
Work instruction: {dynamic_instruction}
Requirements:
1. Identify greenwashing patterns in the text.
2. Quantify greenwashing risk level.
3. Generate greenwashing detection task."""
        else:
            task_description = """Task: Generate a greenwashing detection task (instruction-answer pair) from the text.
Requirements:
1. Identify greenwashing patterns (refer to common patterns list).
2. Quantify greenwashing risk (0-1, 1 = highest risk).
3. Provide evidence chain (quote source text).
4. Produce greenwashing risk report."""
        prompt = f"""You are a senior ESG greenwashing detection expert, skilled at identifying misleading or exaggerated claims in reports.
{task_description}
{greenwashing_patterns}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [detection instruction: identify greenwashing patterns]
Answer: [detection result: matched patterns, risk level, evidence chain]
Output only instruction and answer."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse greenwashing detection response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Detect greenwashing risk in the following text."
        analysis = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("回答：") or line.startswith("回答:") or line.strip().lower().startswith("answer:"):
                analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not analysis:
            analysis = response.strip()
        return ExpertOutput(
            expert_name="greenwashing_detection_expert",
            instruction=instruction,
            response=analysis,
            quality_score=0.0,
            metadata={"expert": "greenwashing_detection_expert", "raw_response": response}
        )


class StandardAlignmentExpert(BaseExpert):
    """Standard alignment expert: ESG disclosure standard identification and compliance."""
    def __init__(self, **kwargs):
        super().__init__(name="standard_alignment_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build standard alignment prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        standards_list = """
Common ESG standards:
- GRI (Global Reporting Initiative)
- TCFD (Task Force on Climate-related Financial Disclosures)
- SASB (Sustainability Accounting Standards Board)
- SSE/SZSE ESG guidelines
- ISO 14001 (Environmental Management)
- CDP (Carbon Disclosure Project)"""
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate a standard alignment task.
Work instruction: {dynamic_instruction}
Requirements:
1. Identify standards followed in the text.
2. Detect mixed use of standards across sections.
3. Generate standard alignment task."""
        else:
            task_description = """Task: Generate a standard alignment task (instruction-answer pair) from the text.
Requirements:
1. Identify ESG standards (GRI, TCFD, SASB, etc.) followed.
2. Detect mixed standards (different sections using different standards).
3. Produce compliance report and identify missing requirements."""
        prompt = f"""You are a senior ESG standard alignment expert, familiar with disclosure standards and regulatory requirements.
{task_description}
{standards_list}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [alignment instruction: identify standards and compliance]
Answer: [alignment result: standards identified, compliance assessment, gaps]
Output only instruction and answer."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse standard alignment response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Identify ESG standards followed in the text and assess compliance."
        analysis = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("回答：") or line.startswith("回答:") or line.strip().lower().startswith("answer:"):
                analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not analysis:
            analysis = response.strip()
        return ExpertOutput(
            expert_name="standard_alignment_expert",
            instruction=instruction,
            response=analysis,
            quality_score=0.0,
            metadata={"expert": "standard_alignment_expert", "raw_response": response}
        )


class KnowledgeGraphExpert(BaseExpert):
    """Knowledge graph expert: entity and relation extraction."""
    def __init__(self, **kwargs):
        super().__init__(name="knowledge_graph_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build knowledge graph prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate a knowledge graph construction task.
Work instruction: {dynamic_instruction}
Requirements:
1. Extract ESG entities.
2. Identify entity relations.
3. Generate knowledge graph task."""
        else:
            task_description = """Task: Generate a knowledge graph construction task (instruction-answer pair) from the text.
Requirements:
1. Extract ESG entities (company, commitments, data, actions, verification).
2. Identify relations (commitment→action, action→data, data→verification).
3. Detect broken links (commitment without action, action without data).
4. Produce entity-relation graph."""
        prompt = f"""You are a senior ESG knowledge graph expert, skilled at extracting entities and relations from text.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [graph construction instruction: extract entities and relations]
Answer: [graph result: entity list, relation list, completeness assessment]
Output only instruction and answer."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse knowledge graph response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Extract ESG entities and relations from the text and build a knowledge graph."
        analysis = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("回答：") or line.startswith("回答:") or line.strip().lower().startswith("answer:"):
                analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not analysis:
            analysis = response.strip()
        return ExpertOutput(
            expert_name="knowledge_graph_expert",
            instruction=instruction,
            response=analysis,
            quality_score=0.0,
            metadata={"expert": "knowledge_graph_expert", "raw_response": response}
        )


class ConsistencyVerificationExpert(BaseExpert):
    """Consistency verification expert: data consistency and contradiction detection."""
    def __init__(self, **kwargs):
        super().__init__(name="consistency_verification_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build consistency verification prompt."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate a consistency verification task.
Work instruction: {dynamic_instruction}
Requirements:
1. Check data consistency.
2. Identify contradictions.
3. Generate consistency verification task."""
        else:
            task_description = """Task: Generate a consistency verification task (instruction-answer pair) from the text.
Requirements:
1. Check whether the same data is stated consistently across the text.
2. Identify contradictions (e.g. scope-1 emissions differing across sections).
3. Detect "missing data" patterns (only favorable data disclosed).
4. Produce consistency report."""
        prompt = f"""You are a senior ESG consistency verification expert, skilled at detecting data contradictions and gaps in reports.
{task_description}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [verification instruction: check data consistency]
Answer: [verification result: consistency assessment, contradictions, missing patterns]
Output only instruction and answer."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse consistency verification response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Verify data consistency in the following text."
        analysis = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("回答：") or line.startswith("回答:") or line.strip().lower().startswith("answer:"):
                analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not analysis:
            analysis = response.strip()
        return ExpertOutput(
            expert_name="consistency_verification_expert",
            instruction=instruction,
            response=analysis,
            quality_score=0.0,
            metadata={"expert": "consistency_verification_expert", "raw_response": response}
        )


class PromisePerformanceVerificationExpert(BaseExpert):
    """Promise-vs-performance verification expert: claim-evidence consistency (tool-use)."""
    def __init__(self, **kwargs):
        super().__init__(name="promise_performance_verification_expert", **kwargs)
    def get_prompt_template(self, chunk_text: str, context: Optional[Dict] = None, dynamic_instruction: Optional[str] = None) -> str:
        """Build promise-performance verification prompt (includes tool-call instructions)."""
        knowledge_context = ""
        if context and context.get("knowledge_items"):
            knowledge_context = "\nRelevant ESG domain knowledge:\n"
            for item in context["knowledge_items"][:3]:
                knowledge_context += f"- {item.get('content', '')}\n"
        tool_use_instruction = """
[Tool-use capability]
This task requires tool-use ability:
1. Identify "Verifiable Claims" in the text.
2. Generate tool-call instructions, e.g.:
   - query_energy_data('Plant A', 2024)
   - query_emission_data('Company', 2024)
   - query_production_data('Plant A', 2024)
   - query_external_verification('claim text')
3. Produce a claim-evidence consistency report from tool returns."""
        if dynamic_instruction:
            task_description = f"""Task: Given the work instruction and the text, generate a promise-performance verification task (with tool calls).
Work instruction: {dynamic_instruction}
Requirements:
1. Extract key claims from the text.
2. Identify verifiable claims.
3. Generate tool-call instructions.
4. Generate consistency verification task."""
        else:
            task_description = """Task: Generate a promise-performance verification task (instruction-answer pair) with tool-use.
Requirements:
1. Extract key claims (e.g. "Plant A reduced unit energy use by 15%").
2. Identify verifiable claims (with external data).
3. Generate tool-call instructions (e.g. query_energy_data('Plant A', 2024)).
4. Produce claim-evidence consistency report: claim, tool calls, expected data type, consistency level (high/medium exaggeration/severe mismatch)."""
        prompt = f"""You are a senior ESG promise-performance verification expert, skilled at identifying verifiable claims and generating tool-call instructions.
{task_description}
{tool_use_instruction}
{knowledge_context}
Text:
{chunk_text}
Output format:
Instruction: [verification instruction: identify claims and generate tool calls]
Answer: [result: claims, tool-call instructions, expected data, consistency criteria]
The answer must include tool-call instructions (e.g. query_energy_data('Plant A', 2024)).
Output only instruction and answer."""
        return prompt
    def parse_response(self, response: str) -> ExpertOutput:
        """Parse promise-performance verification response (supports Chinese/English labels)."""
        lines = response.strip().split('\n')
        instruction = "Identify verifiable claims in the text and generate tool-call instructions for verification."
        analysis = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.strip().lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("回答：") or line.startswith("回答:") or line.strip().lower().startswith("answer:"):
                analysis = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not analysis:
            analysis = response.strip()
        return ExpertOutput(
            expert_name="promise_performance_verification_expert",
            instruction=instruction,
            response=analysis,
            quality_score=0.0,
            metadata={
                "expert": "promise_performance_verification_expert",
                "raw_response": response,
                "requires_tool_use": True
            }
        )


def create_expert(expert_name: str, config: Dict) -> BaseExpert:
    """Create expert instance by name. config: api_provider, model_name, temperature, max_tokens."""
    expert_classes = {
        "qa_expert": QAExpert,
        "summary_expert": SummaryExpert,
        "extraction_expert": ExtractionExpert,
        "classification_expert": ClassificationExpert,
        "analysis_expert": AnalysisExpert,
        "temporal_analysis_expert": TemporalAnalysisExpert,
        "benchmark_comparison_expert": BenchmarkComparisonExpert,
        "greenwashing_detection_expert": GreenwashingDetectionExpert,
        "standard_alignment_expert": StandardAlignmentExpert,
        "knowledge_graph_expert": KnowledgeGraphExpert,
        "consistency_verification_expert": ConsistencyVerificationExpert,
        "promise_performance_verification_expert": PromisePerformanceVerificationExpert
    }
    if expert_name not in expert_classes:
        raise ValueError(f"Unknown expert type: {expert_name}")
    
    expert_class = expert_classes[expert_name]
    return expert_class(
        api_provider=config.get("api_provider", "openai"),
        model_name=config.get("model_name", "gpt-4"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 2048)
    )

