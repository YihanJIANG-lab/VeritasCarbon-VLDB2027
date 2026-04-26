"""
02-02 Domain knowledge injection.

Retrieve ESG domain knowledge and inject into instruction generation.
"""

import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class DomainKnowledgeInjector:
    """
    Domain knowledge injector: retrieve ESG knowledge to enhance instruction generation.
    """
    ESG_KEYWORDS = {
        "环境": ["碳排放", "温室气体", "能源", "水资源", "废弃物", "污染", "环保", "绿色"],
        "社会": ["员工", "培训", "安全", "健康", "供应链", "社区", "公益", "社会责任"],
        "治理": ["公司治理", "合规", "风险", "内部控制", "董事会", "审计", "反腐败"]
    }
    
    GRI_KEYWORDS = [
        "GRI", "实质性议题", "利益相关方", "可持续发展", "ESG指标",
        "环境绩效", "社会绩效", "治理绩效"
    ]
    
    SDG_KEYWORDS = [
        "可持续发展目标", "SDG", "消除贫困", "零饥饿", "良好健康",
        "优质教育", "性别平等", "清洁能源", "体面工作", "产业创新"
    ]
    
    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        max_knowledge_items: int = 5
    ):
        """
        Args:
            knowledge_base_path: Path to knowledge base.
            max_knowledge_items: Max number of knowledge items to inject.
        """
        self.knowledge_base_path = Path(knowledge_base_path) if knowledge_base_path else None
        self.max_knowledge_items = max_knowledge_items
        self.knowledge_base = None
        if self.knowledge_base_path and self.knowledge_base_path.exists():
            self._load_knowledge_base()
        else:
            logger.warning("Knowledge base path not found; using built-in keywords.")
    
    def _load_knowledge_base(self):
        """Load knowledge base from disk."""
        try:
            entities_file = self.knowledge_base_path / "esg_entities.jsonl"
            relations_file = self.knowledge_base_path / "esg_relations.jsonl"
            
            self.knowledge_base = {
                "entities": [],
                "relations": []
            }
            
            if entities_file.exists():
                with open(entities_file, "r", encoding="utf-8") as f:
                    for line in f:
                        entity = json.loads(line)
                        self.knowledge_base["entities"].append(entity)
            
            if relations_file.exists():
                with open(relations_file, "r", encoding="utf-8") as f:
                    for line in f:
                        relation = json.loads(line)
                        self.knowledge_base["relations"].append(relation)
            
            logger.info(f"Loaded knowledge base: {len(self.knowledge_base['entities'])} entities, "
                       f"{len(self.knowledge_base['relations'])} relations")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            self.knowledge_base = None
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract ESG keywords from text. Returns list of keywords."""
        keywords = []
        text_lower = text.lower()
        for dimension, dim_keywords in self.ESG_KEYWORDS.items():
            for kw in dim_keywords:
                if kw in text_lower:
                    keywords.append(f"{dimension}:{kw}")
        
        for kw in self.GRI_KEYWORDS:
            if kw in text_lower:
                keywords.append(f"GRI:{kw}")
        for kw in self.SDG_KEYWORDS:
            if kw in text_lower:
                keywords.append(f"SDG:{kw}")
        return list(set(keywords))
    
    def retrieve_knowledge(self, chunk_text: str) -> List[Dict[str, str]]:
        """Retrieve relevant knowledge for chunk. Returns list of items with type and content."""
        knowledge_items = []
        keywords = self.extract_keywords(chunk_text)
        if self.knowledge_base:
            for entity in self.knowledge_base["entities"]:
                entity_text = entity.get("text", "").lower()
                if any(kw.split(":")[-1] in entity_text for kw in keywords):
                    knowledge_items.append({
                        "type": "entity",
                        "content": f"Entity: {entity.get('name', '')} - {entity.get('description', '')}",
                        "source": "knowledge_base"
                    })
            for relation in self.knowledge_base["relations"]:
                relation_text = str(relation).lower()
                if any(kw.split(":")[-1] in relation_text for kw in keywords):
                    knowledge_items.append({
                        "type": "relation",
                        "content": f"Relation: {relation}",
                        "source": "knowledge_base"
                    })
        for kw in keywords[:self.max_knowledge_items]:
            dimension, keyword = kw.split(":") if ":" in kw else ("general", kw)
            knowledge_items.append({
                "type": "keyword",
                "content": f"ESG keyword: {keyword} (dimension: {dimension})",
                "source": "builtin"
            })
        if any("gri" in text.lower() or "实质性" in text for text in [chunk_text]):
            knowledge_items.append({
                "type": "standard",
                "content": "GRI: Global Reporting Initiative sustainability reporting standards",
                "source": "builtin"
            })
        return knowledge_items[:self.max_knowledge_items]
    
    def inject_knowledge_to_prompt(
        self,
        base_prompt: str,
        chunk_text: str,
        knowledge_items: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Inject domain knowledge into prompt. If knowledge_items is None, retrieve automatically."""
        if knowledge_items is None:
            knowledge_items = self.retrieve_knowledge(chunk_text)
        if not knowledge_items:
            return base_prompt
        knowledge_context = "Relevant ESG domain knowledge:\n"
        for i, item in enumerate(knowledge_items, 1):
            knowledge_context += f"{i}. {item['content']}\n"
        enhanced_prompt = f"""{base_prompt}

{knowledge_context}

Generate high-quality instruction-answer pairs based on the above ESG domain knowledge."""
        return enhanced_prompt
    
    def get_knowledge_summary(self, chunk_text: str) -> str:
        """Get a short summary of retrieved knowledge (for logging/debug)."""
        knowledge_items = self.retrieve_knowledge(chunk_text)
        if not knowledge_items:
            return "No relevant domain knowledge"
        summary = f"Retrieved {len(knowledge_items)} knowledge items:\n"
        for item in knowledge_items:
            summary += f"  - {item['type']}: {item['content'][:50]}...\n"
        
        return summary

