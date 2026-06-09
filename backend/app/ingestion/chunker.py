import re
import hashlib
from typing import List, Dict, Any, Optional

class DocumentChunker:
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

    def chunk_document(self, parsed_doc, spans) -> list:
        from app.models.schemas import DocumentChunk, ChunkMetadata, ChunkType, ContentType
        all_chunks = []
        
        for span in spans:
            text = span.full_text
            if not text.strip():
                continue
                
            parent_id = f"parent_{self.generate_hash(text)}"
            
            base_metadata = {
                "document_id": parsed_doc.document_id,
                "book_name": parsed_doc.book_name,
                "utility_domain": parsed_doc.utility_domain,
                "chapter_num": span.chapter_num,
                "chapter_title": span.chapter_title,
                "section_title": span.section_title,
                "subsection_title": span.subsection_title,
                "page_start": span.page_start,
                "page_end": span.page_end,
            }
            
            raw_chunks = self.split_into_semantic_chunks(text, parent_id, base_metadata)
            
            for rc in raw_chunks:
                meta_dict = rc["metadata"]
                chunk_type_str = meta_dict.get("chunk_type")
                
                ct = ChunkType.SEMANTIC
                ctype = ContentType.TEXT
                if chunk_type_str == "table_chunk":
                    ct = ChunkType.TABLE
                    ctype = ContentType.TABLE
                elif chunk_type_str == "formula_chunk":
                    ct = ChunkType.FORMULA
                    ctype = ContentType.FORMULA
                    
                meta_obj = ChunkMetadata(
                    chunk_id=rc["chunk_id"],
                    document_id=parsed_doc.document_id,
                    book_name=parsed_doc.book_name,
                    utility_domain=parsed_doc.utility_domain,
                    chapter_num=span.chapter_num,
                    chapter_title=span.chapter_title,
                    section_title=span.section_title,
                    subsection_title=span.subsection_title,
                    page_start=span.page_start,
                    page_end=span.page_end,
                    chunk_type=ct,
                    content_type=ctype,
                    word_count=len(rc["text"].split()),
                    char_count=len(rc["text"])
                )
                all_chunks.append(DocumentChunk(text=rc["text"], metadata=meta_obj))
                
        return all_chunks

chunker = DocumentChunker()
