import re
import hashlib
from typing import List, Dict, Any, Optional

class Chunker:
    def __init__(self, semantic_size=500, semantic_overlap=100):
        self.semantic_size = semantic_size
        self.semantic_overlap = semantic_overlap

    def generate_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def split_into_semantic_chunks(self, text: str, parent_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.semantic_size - self.semantic_overlap):
            chunk_text = " ".join(words[i:i + self.semantic_size])
            if len(chunk_text.strip()) == 0:
                continue
            
            chunk_id = f"semantic_{self.generate_hash(chunk_text)}"
            
            has_table = "table" in chunk_text.lower()
            has_formula = bool(re.search(r'(\b\w+\s*=\s*[^a-zA-Z]|\bη\b|\b%\b)', chunk_text))
            
            chunks.append({
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_type": "semantic_chunk",
                    "has_table": has_table,
                    "has_formula": has_formula
                }
            })
            
            # Additional table chunk if table is clearly detected
            if has_table and "table " in chunk_text.lower():
                chunks.append({
                    "chunk_id": f"table_{self.generate_hash(chunk_text)}",
                    "parent_id": parent_id,
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_type": "table_chunk",
                        "has_table": True,
                        "has_formula": False
                    }
                })
                
            # Additional formula chunk if formula is detected
            if has_formula:
                chunks.append({
                    "chunk_id": f"formula_{self.generate_hash(chunk_text)}",
                    "parent_id": parent_id,
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_type": "formula_chunk",
                        "has_table": False,
                        "has_formula": True
                    }
                })
                
        return chunks

    def process_document(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a document divided into sections (e.g., from OCR).
        Input section format:
        {
            "book_name": "...",
            "utility_domain": "...",
            "chapter_title": "...",
            "section_title": "...",
            "page_start": 0,
            "page_end": 0,
            "text": "..."
        }
        """
        all_chunks = []
        
        for section in sections:
            parent_text = section.get("text", "")
            if not parent_text:
                continue
                
            parent_id = f"parent_{self.generate_hash(parent_text)}"
            
            base_metadata = {
                "book_name": section.get("book_name", ""),
                "utility_domain": section.get("utility_domain", "unknown"),
                "chapter_title": section.get("chapter_title", ""),
                "section_title": section.get("section_title", ""),
                "page_start": section.get("page_start", 0),
                "page_end": section.get("page_end", 0),
                "source_hash": self.generate_hash(parent_text)
            }
            
            # 1. Add the parent chunk itself
            all_chunks.append({
                "chunk_id": parent_id,
                "parent_id": parent_id,
                "text": parent_text,
                "metadata": {
                    **base_metadata,
                    "chunk_type": "parent_section",
                    "has_table": False,
                    "has_formula": False
                }
            })
            
            # 2. Add semantic and specific (table/formula) child chunks
            child_chunks = self.split_into_semantic_chunks(parent_text, parent_id, base_metadata)
            all_chunks.extend(child_chunks)
            
        return all_chunks

chunker = Chunker()
