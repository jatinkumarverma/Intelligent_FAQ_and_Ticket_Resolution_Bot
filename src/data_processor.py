"""
Data Processing Module for Support Bot

Handles:
- Loading historical tickets from CSV
- Parsing product documentation (PDF, DOCX, TXT, MD)
- Text cleaning and normalization
- Document chunking with overlap
- Metadata generation
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json

import pandas as pd
from tqdm import tqdm

from src.config import (
    DOCS_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    SUPPORTED_DOC_TYPES, MAX_CHUNKS_PER_DOC
)
from src.logger import logger

# Try to import optional libraries
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    logger.warning("PyPDF2 not installed. PDF parsing disabled.")

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    logger.warning("python-docx not installed. DOCX parsing disabled.")


@dataclass
class TicketRecord:
    """Historical support ticket record."""
    id: str
    customer_name: str
    customer_email: str
    customer_age: str
    customer_gender: str
    product_purchased: str
    purchase_date: str
    ticket_type: str
    subject: str
    description: str
    status: str
    resolution: str
    priority: str
    channel: str
    first_response_time: str
    time_to_resolution: str
    satisfaction_rating: str
    
    def __hash__(self):
        return hash(self.id)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'subject': self.subject,
            'description': self.description,
            'category': self.category,
            'resolution': self.resolution,
            'timestamp': self.timestamp,
        }


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    text: str
    source: str  # Original document name
    doc_type: str  # Type: 'ticket', 'documentation', etc.
    chunk_id: str  # Unique identifier
    metadata: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.chunk_id)


class DataProcessor:
    """Processes and prepares data for knowledge base ingestion."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logger
        self.tickets: List[TicketRecord] = []
        self.documents: List[DocumentChunk] = []
        self.stats = {
            'tickets_loaded': 0,
            'documents_loaded': 0,
            'chunks_created': 0,
            'errors': 0,
        }
    
    # ============= TICKET LOADING =============
    
    def load_tickets_csv(self, filepath: str) -> List[TicketRecord]:
        """
        Load historical tickets from CSV file.
        
        Expected columns: id, customer_id, subject, description, category, resolution, timestamp
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of TicketRecord objects
        """
        self.logger.info(f"Loading tickets from {filepath}")
        
        if not os.path.exists(filepath):
            self.logger.error(f"CSV file not found: {filepath}")
            return []
        
        tickets = []
        
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = {
                            'Ticket ID',
                            'Customer Name',
                            'Customer Email',
                            'Ticket Subject',
                            'Ticket Description',
                            'Ticket Type',
                            'Resolution'
                            }
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                self.stats['errors'] += 1
                return []
            
            # Convert rows to TicketRecord objects
            for idx, row in df.iterrows():
                try:
                    ticket = TicketRecord(
                        id=str(row['Ticket ID']),
                        customer_name=str(row['Customer Name']),
                        customer_email=str(row['Customer Email']),
                        customer_age=str(row['Customer Age']),
                        customer_gender=str(row['Customer Gender']),
                        product_purchased=str(row['Product Purchased']),
                        purchase_date=str(row['Date of Purchase']),
                        ticket_type=str(row['Ticket Type']),
                        subject=str(row['Ticket Subject']),
                        description=str(row['Ticket Description']),
                        status=str(row['Ticket Status']),
                        resolution=str(row['Resolution']),
                        priority=str(row['Ticket Priority']),
                        channel=str(row['Ticket Channel']),
                        first_response_time=str(row['First Response Time']),
                        time_to_resolution=str(row['Time to Resolution']),
                        satisfaction_rating=str(row['Customer Satisfaction Rating']),
                    )
                    tickets.append(ticket)
                except Exception as e:
                    self.logger.warning(f"Error processing row {idx}: {e}")
                    self.stats['errors'] += 1
                    continue
            
            self.tickets = tickets
            self.stats['tickets_loaded'] = len(tickets)
            self.logger.info(f"Successfully loaded {len(tickets)} tickets")
            return tickets
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            self.stats['errors'] += 1
            return []
    
    # ============= DOCUMENT LOADING =============
    
    def load_documentation(self, doc_dir: str) -> List[str]:
        """
        Load all documentation files from a directory.
        
        Supported formats: .pdf, .docx, .txt, .md
        
        Args:
            doc_dir: Directory containing documentation files
            
        Returns:
            List of extracted text from documents
        """
        self.logger.info(f"Loading documentation from {doc_dir}")
        
        if not os.path.isdir(doc_dir):
            self.logger.error(f"Documentation directory not found: {doc_dir}")
            return []
        
        documents = []
        doc_path = Path(doc_dir)
        
        # Find all supported files
        for file_path in tqdm(doc_path.rglob('*'), desc="Scanning docs"):
            if not file_path.is_file():
                continue
            
            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_DOC_TYPES:
                continue
            
            try:
                if suffix == '.pdf':
                    text = self._parse_pdf(file_path)
                elif suffix == '.docx':
                    text = self._parse_docx(file_path)
                elif suffix in {'.txt', '.md'}:
                    text = self._parse_text(file_path)
                else:
                    continue
                
                if text and text.strip():
                    documents.append({
                        'source': file_path.name,
                        'path': str(file_path),
                        'type': suffix[1:],
                        'text': text,
                    })
                    self.stats['documents_loaded'] += 1
                    
            except Exception as e:
                self.logger.warning(f"Error processing {file_path.name}: {e}")
                self.stats['errors'] += 1
                continue
        
        self.logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _parse_pdf(self, filepath: Path) -> str:
        """Extract text from PDF file."""
        if not HAS_PDF:
            self.logger.warning(f"PyPDF2 not available. Skipping {filepath.name}")
            return ""
        
        try:
            text = []
            with open(filepath, 'rb') as f:
                reader = PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text.append(page_text)
                    except Exception as e:
                        self.logger.debug(f"Error extracting page {page_num}: {e}")
            return "\n".join(text)
        except Exception as e:
            self.logger.warning(f"Error parsing PDF {filepath.name}: {e}")
            return ""
    
    def _parse_docx(self, filepath: Path) -> str:
        """Extract text from DOCX file."""
        if not HAS_DOCX:
            self.logger.warning(f"python-docx not available. Skipping {filepath.name}")
            return ""
        
        try:
            doc = Document(filepath)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            self.logger.warning(f"Error parsing DOCX {filepath.name}: {e}")
            return ""
    
    def _parse_text(self, filepath: Path) -> str:
        """Extract text from TXT/MD file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"Error parsing text file {filepath.name}: {e}")
            return ""
        
    def _parse_json(self, filepath: Path) -> str:
        """Extract text from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert JSON to readable text
                return self._json_to_text(data)

        except Exception as e:
            self.logger.warning(f"Error parsing JSON {filepath.name}: {e}")
            return ""
        
    def _json_to_text(self, obj, prefix="") -> str:
        """Convert nested JSON into searchable text."""
        lines = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                lines.append(self._json_to_text(value, new_prefix))

        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                lines.append(self._json_to_text(item, f"{prefix}[{idx}]"))

        else:
           lines.append(f"{prefix}: {obj}")

        return "\n".join(filter(None, lines))
    
    # ============= TEXT NORMALIZATION =============
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common artifacts
        text = text.replace('\x00', '')  # Null bytes
        text = text.replace('\ufeff', '')  # BOM
        
        return text.strip()
    
    # ============= DOCUMENT CHUNKING =============
    
    def chunk_documents(
        self,
        documents: List[Dict],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> List[DocumentChunk]:
        """
        Split documents into overlapping chunks.
        
        Args:
            documents: List of document dicts with 'text', 'source', 'type'
            chunk_size: Tokens per chunk (approximate, based on word count)
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of DocumentChunk objects
        """
        self.logger.info(f"Chunking {len(documents)} documents (size={chunk_size}, overlap={chunk_overlap})")
        
        chunks = []
        chunk_counter = 0
        
        for doc in tqdm(documents, desc="Chunking documents"):
            text = self.normalize_text(doc['text'])
            source = doc['source']
            doc_type = doc.get('type', 'unknown')
            
            # Split into sentences for better chunking
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                continue
            
            # Create chunks from sentences
            current_chunk = []
            current_tokens = 0
            chunk_start_idx = 0
            
            for sent_idx, sentence in enumerate(sentences):
                # Approximate tokens as words
                sent_tokens = len(sentence.split())
                
                # Check if adding this sentence would exceed chunk size
                if current_tokens + sent_tokens > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_id = f"{source}_{chunk_counter}"
                    
                    chunk = DocumentChunk(
                        text=chunk_text,
                        source=source,
                        doc_type=doc_type,
                        chunk_id=chunk_id,
                        metadata={
                            'start_sentence': chunk_start_idx,
                            'end_sentence': sent_idx,
                            'source_type': doc_type,
                            'created_at': datetime.now().isoformat(),
                        }
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                    
                    # Handle overlap by keeping last sentences
                    overlap_sentences = self._calculate_overlap_sentences(
                        current_chunk, chunk_overlap
                    )
                    current_chunk = overlap_sentences
                    current_tokens = sum(len(s.split()) for s in overlap_sentences)
                    chunk_start_idx = max(0, sent_idx - len(overlap_sentences))
                
                current_chunk.append(sentence)
                current_tokens += sent_tokens
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{source}_{chunk_counter}"
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    source=source,
                    doc_type=doc_type,
                    chunk_id=chunk_id,
                    metadata={
                        'is_last': True,
                        'source_type': doc_type,
                        'created_at': datetime.now().isoformat(),
                    }
                )
                chunks.append(chunk)
                chunk_counter += 1
        
        self.documents = chunks
        self.stats['chunks_created'] = len(chunks)
        self.logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Simple sentence splitting by common punctuation."""
        # Basic splitting on sentence endings
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def _calculate_overlap_sentences(
        sentences: List[str],
        overlap_tokens: int
    ) -> List[str]:
        """Calculate which sentences to keep for overlap."""
        current_tokens = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            sent_tokens = len(sentence.split())
            if current_tokens + sent_tokens > overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            current_tokens += sent_tokens
        
        return overlap_sentences
    
    # ============= TICKET CHUNKING =============
    
    def chunk_tickets(self) -> List[DocumentChunk]:
        """
        Convert historical tickets into chunks for knowledge base.
        
        Returns:
            List of DocumentChunk objects
        """
        self.logger.info(f"Chunking {len(self.tickets)} historical tickets")
        
        chunks = []
        
        for idx, ticket in enumerate(self.tickets):
            # Combine relevant fields
            text = f"""
            Ticket ID: {ticket.id}
            Customer: {ticket.customer_name}
            Product: {ticket.product_purchased}
            Ticket Type: {ticket.ticket_type}
            Priority: {ticket.priority}
            Channel: {ticket.channel}

            Subject: {ticket.subject}

            Description:
            {ticket.description}

            Resolution:
            {ticket.resolution}

            Status: {ticket.status}
            Customer Satisfaction: {ticket.satisfaction_rating}
            """.strip()
            
            text = self.normalize_text(text)
            
            chunk = DocumentChunk(
                text=text,
                source=f"ticket_{ticket.id}",
                doc_type="ticket",
                chunk_id=f"ticket_{ticket.id}",
                metadata={
                    'ticket_id': ticket.id,
                    'customer_id': ticket.customer_id,
                    'category': ticket.category,
                    'timestamp': ticket.timestamp,
                    'source_type': 'historical_ticket',
                }
            )
            chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} chunks from historical tickets")
        return chunks
    
    # ============= COMBINED PIPELINE =============
    
    def process_all_data(
        self,
        tickets_csv: Optional[str] = None,
        docs_dir: Optional[str] = None,
    ) -> Tuple[List[DocumentChunk], List[TicketRecord]]:
        """
        Complete pipeline: load tickets + docs, chunk them, return combined chunks.
        
        Args:
            tickets_csv: Path to tickets CSV (if None, uses data/tickets.csv)
            docs_dir: Path to docs directory (if None, uses data/docs)
            
        Returns:
            Tuple of (all_chunks, tickets)
        """
        if tickets_csv is None:
            tickets_csv = str(DATA_DIR / "tickets.csv")
        if docs_dir is None:
            docs_dir = str(DOCS_DIR)
        
        self.logger.info("Starting complete data processing pipeline")
        
        # Load tickets
        self.load_tickets_csv(tickets_csv)
        
        # Load and chunk documentation
        doc_list = self.load_documentation(docs_dir)
        doc_chunks = self.chunk_documents(doc_list) if doc_list else []
        
        # Chunk historical tickets
        ticket_chunks = self.chunk_tickets()
        
        # Combine all chunks
        all_chunks = doc_chunks + ticket_chunks
        
        self.logger.info(f"Pipeline complete: {len(all_chunks)} total chunks")
        self.logger.info(f"Stats: {self.stats}")
        
        return all_chunks, self.tickets
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()


def main():
    """Example usage."""
    processor = DataProcessor()
    
    # Process all data
    chunks, tickets = processor.process_all_data()
    
    print(f"\n✅ Processing Complete!")
    print(f"  - Loaded {len(tickets)} historical tickets")
    print(f"  - Created {len(chunks)} chunks")
    print(f"  - Stats: {processor.get_stats()}")
    
    # Show sample chunks
    if chunks:
        print(f"\n📚 Sample chunks:")
        for chunk in chunks[:3]:
            print(f"  - {chunk.source} ({chunk.doc_type}): {chunk.text[:100]}...")


if __name__ == "__main__":
    main()
