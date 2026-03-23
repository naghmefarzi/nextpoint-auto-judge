#!/usr/bin/env python3
"""
Convert NextPoint deposition transcripts to auto-judge-base Report format.

Steps:
  1. Load deposition transcript file(s)
  2. Extract Q&A pairs using the existing QAExtractor (no AWS required)
  3. Chunk pairs into overlapping passages
  4. Convert each chunk to a trec_auto_judge Document object
  5. Create a fake Report: all chunks in `documents`, one sentence citing them all
  6. Write with write_pydantic_json_list

Usage (run from src/):
    python deposition_to_autojudge_report.py \
        --input ../Summarization-SHARED-EXTERNALLY/data/001-Original9/transcripts/text/20240102/*.txt \
        --output-dir ../results/autojudge_reports/ \
        --query-id nextpoint-deposition \
        --chunk-size 4000 \
        --overlap 2
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trec_auto_judge.document.document import Document
from trec_auto_judge.report import (
    NeuclirReportSentence,
    Report,
    ReportMetaData,
    write_pydantic_json_list,
)


# ---------------------------------------------------------------------------
# Minimal transcript reader (inlined to avoid the boto3 import chain)
# ---------------------------------------------------------------------------

def read_transcript_file(filepath: str) -> List[str]:
    """Read a deposition transcript, ignoring encoding errors."""
    with open(filepath, "r", encoding="utf-8", newline="", errors="ignore") as f:
        return f.readlines()


# ---------------------------------------------------------------------------
# Minimal QA extractor (parsing-only subset of QAExtractor, no AWS required)
# ---------------------------------------------------------------------------

class _QAExtractor:
    """Parse Q/A pairs from a deposition transcript (no LLM calls)."""

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self.qa_pairs: List[Tuple] = []
        self.introductory_lines: List[str] = []
        self.in_intro = True
        self.current_question: List[str] = []
        self.current_answer: List[str] = []
        self.mode: Optional[str] = None
        self.current_page = 0
        self.question_line_number = 0
        self.question_page_number = 0
        self.answer_line_number = 0
        self.answer_page_number = 0
        self.waiting_for_page_number = False
        self.orphan_lines: List[str] = []  # non-Q/A lines encountered after intro

    def extract_qa_pairs(self, lines: List[str]) -> Tuple[List[Tuple], List[str], List[str]]:
        self._reset_state()
        for line in lines:
            if self._handle_page_number(line):
                continue
            self._process_line(line)
        self._flush_current_pair()
        return self.qa_pairs, self.introductory_lines, self.orphan_lines

    def _handle_page_number(self, line: str) -> bool:
        page_related = False
        if "\f" in line:
            self.waiting_for_page_number = True
            page_related = True
            after_f = line.split("\f", 1)[-1]
            m = re.search(r"(\d+)", after_f)
            if m:
                self.current_page = int(m.group(1))
                self.waiting_for_page_number = False
        elif self.waiting_for_page_number:
            if re.match(r"^\s*\d+\s+(Q|A|\w)", line.strip()):
                self.waiting_for_page_number = False
                return False
            page_related = True
            m = re.search(r"(\d+)", line)
            if m:
                self.current_page = int(m.group(1))
                self.waiting_for_page_number = False
        if page_related and self.in_intro:
            self.introductory_lines.append(line.strip())
        return page_related

    def _process_line(self, line: str):
        if self.in_intro:
            if self._is_q(line):
                self.in_intro = False
                self._start_question(line)
            else:
                self.introductory_lines.append(line)
        elif self._is_q(line):
            if self.mode == "A" or not self.current_question:
                self._start_question(line)
            else:
                self.current_question.append(re.sub(r"^\s*\d+\s+Q\b\s*", "", line.strip()))
        elif self._is_a(line):
            self._start_answer(line)
        elif re.match(r"^\s*\d+\s+", line.strip()):
            content = re.sub(r"^\s*\d+\s+", "", line.strip())
            if content:
                (self.current_question if self.mode == "Q" else self.current_answer).append(content)
        else:
            stripped = line.strip()
            if stripped:
                self.orphan_lines.append(stripped)

    def _is_q(self, line: str) -> bool:
        return bool(re.match(r"^\s*\d+[:.\s]+\s*Q\b", line.strip()))

    def _is_a(self, line: str) -> bool:
        return bool(re.match(r"^\s*\d+[:.\s]+\s*A\b", line.strip()))

    def _start_question(self, line: str):
        self._flush_current_pair()
        m = re.match(r"^\s*(\d+)", line.strip())
        if m:
            self.question_line_number = int(m.group(1))
        self.question_page_number = self.current_page
        self.current_question = [re.sub(r"^\s*\d+\s+Q\b\s*", "", line.strip())]
        self.current_answer = []
        self.mode = "Q"

    def _start_answer(self, line: str):
        m = re.match(r"^\s*(\d+)", line.strip())
        if m:
            self.answer_line_number = int(m.group(1))
        self.answer_page_number = self.current_page
        self.current_answer = [re.sub(r"^\s*\d+\s+A\b\s*", "", line.strip())]
        self.mode = "A"

    def _flush_current_pair(self):
        if not self.current_question:
            return
        q = re.sub(r"\s+", " ", " ".join(self.current_question).strip())
        a = re.sub(r"\s+", " ", " ".join(self.current_answer).strip())
        a_page = self.answer_page_number if self.answer_page_number > 0 else self.question_page_number
        a_line = self.answer_line_number if self.answer_line_number > 0 else self.question_line_number
        self.qa_pairs.append((q, a, self.question_page_number, self.question_line_number, a_page, a_line))
        self.current_question = []
        self.current_answer = []
        self.mode = None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_full_text(
    full_text: str,
    answer_ends: List[int],
    chunk_size_chars: int = 4000,
    overlap_chars: int = 200,
) -> List[Tuple[int, int]]:
    """Split full_text into overlapping (start, end) char ranges.

    Chunk boundaries are snapped to positions in answer_ends so that no chunk
    ends in the middle of an answer.  Overlap is character-based.

    Args:
        full_text:        The complete deposition text.
        answer_ends:      Sorted list of char positions where answers end
                          (valid cut points).  Must include len(full_text).
        chunk_size_chars: Soft upper limit on characters per chunk.
        overlap_chars:    Characters of overlap between consecutive chunks.

    Returns:
        List of (start, end) character ranges into full_text.
    """
    if not answer_ends:
        return [(0, len(full_text))]

    ranges: List[Tuple[int, int]] = []
    start = 0

    for end in answer_ends:
        if end - start >= chunk_size_chars:
            ranges.append((start, end))
            start = max(0, end - overlap_chars)

    # Final chunk covering any remaining text
    if start < len(full_text):
        ranges.append((start, len(full_text)))

    return ranges


# ---------------------------------------------------------------------------
# Document creation
# ---------------------------------------------------------------------------

def deposition_to_documents(
    filepath: str,
    depo_id: str,
    chunk_size_chars: int = 4000,
    overlap_chars: int = 200,
) -> List[Document]:
    """Load a deposition and return one Document per chunk plus a fulltext Document.

    The full text (intro + Q&A + trailing content) is stored as a single
    '{depo_id}_fulltext' document.  Chunks are character-based overlapping
    slices of that full text, always ending at the boundary of a complete answer.

    Args:
        filepath:         Path to the deposition .txt file.
        depo_id:          Short identifier used as the doc_id prefix.
        chunk_size_chars: Soft upper limit on characters per chunk.
        overlap_chars:    Characters of overlap between consecutive chunks.

    Returns:
        List of Document objects.
    """
    print(f"  Reading: {filepath}")
    lines = read_transcript_file(filepath)

    # --- Build clean full text directly from raw lines ---
    clean_lines: List[str] = []
    q_line_indices: List[int] = []

    for raw in lines:
        raw = raw.replace("\f", "")
        stripped = raw.strip()
        if not stripped:
            clean_lines.append("")
            continue
        if re.match(r"^\s*\d+[:.\s]+\s*Q\b", stripped):
            q_line_indices.append(len(clean_lines))
        # clean_lines.append(re.sub(r"^\s*\d+\s+", "", stripped))
        clean_lines.append(stripped)

    full_text = "\n".join(clean_lines)
    if not full_text.strip():
        print(f"  WARNING: no content found in {filepath}")
        return []

    print(f"  Full text: {len(full_text)} chars")

    # --- Cut points: end of last answer line just before each new Q line ---
    line_starts: List[int] = []
    pos = 0
    for line in clean_lines:
        line_starts.append(pos)
        pos += len(line) + 1

    answer_ends: List[int] = []
    for qi in q_line_indices:
        if qi == 0:
            continue
        j = qi - 1
        while j >= 0 and not clean_lines[j].strip():
            j -= 1
        if j >= 0:
            cut = line_starts[j] + len(clean_lines[j])
            if not answer_ends or cut != answer_ends[-1]:
                answer_ends.append(cut)
    answer_ends.append(len(full_text))

    # --- Chunk ---
    ranges = chunk_full_text(
        full_text, answer_ends,
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
    )
    print(f"  Chunked into {len(ranges)} passages "
          f"(~{chunk_size_chars} chars each, overlap={overlap_chars} chars)")

    # --- Documents ---
    fulltext_doc_id = f"{depo_id}_fulltext"
    documents: List[Document] = [
        Document(
            id=fulltext_doc_id,
            text=full_text,
            title=f"{depo_id} (complete deposition)",
            metadata={
                "deposition_id": depo_id,
                "chunk_type": "fulltext",
                "source_file": str(filepath),
            },
        )
    ]

    # for i, (ft_start, ft_end) in enumerate(ranges):
    #     documents.append(Document(
    #         id=f"{depo_id}_chunk_{i:04d}",
    #         text=full_text[ft_start:ft_end],
    #         title=f"{depo_id} (chunk {i + 1}/{len(ranges)})",
    #         metadata={
    #             "deposition_id": depo_id,
    #             "chunk_type": "qa",
    #             "chunk_index": i,
    #             "total_chunks": len(ranges),
    #             "source_file": str(filepath),
    #             "fulltext_doc_id": fulltext_doc_id,
    #             "full_text_start": ft_start,
    #             "full_text_end": ft_end,
    #         },
    #     ))

    return documents


# ---------------------------------------------------------------------------
# Report creation
# ---------------------------------------------------------------------------

def create_fake_report(
    documents: List[Document],
    query_id: str,
    run_id: str = "nextpoint-depo",
    team_id: str = "nextpoint",
) -> Report:
    """Build a fake Report: all chunks in `documents`, one sentence citing all.

    Uses NeuclirReportSentence (citations = list of doc_ids) and
    is_ragtime=False so no RAGTIME-specific validation is triggered.
    """
    doc_dict: Dict[str, Document] = {doc.id: doc for doc in documents}
    # Cite only the chunks, not the fulltext document (it's a lookup target, not a result)
    cited_ids: List[str] = [
        doc.id for doc in documents
        if (doc.metadata or {}).get("chunk_type") == "fulltext"
    ]

    fake_sentence = NeuclirReportSentence(
        text=(
            "This report contains the full deposition transcript "
            "split into overlapping chunks."
        ),
        citations=cited_ids,
    )

    metadata = ReportMetaData(
        team_id=team_id,
        run_id=run_id,
        topic_id=query_id,
    )

    return Report(
        is_ragtime=False,
        metadata=metadata,
        responses=[fake_sentence],
        documents=doc_dict,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NextPoint depositions to auto-judge-base Report format."
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more deposition transcript file paths.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where <depo_id>_report.jsonl files will be written.",
    )
    parser.add_argument(
        "--query-id", default="nextpoint-deposition",
        help="Query / topic ID written into report metadata (default: nextpoint-deposition).",
    )
    parser.add_argument(
        "--run-id", default="nextpoint-depo",
        help="Run ID written into report metadata.",
    )
    parser.add_argument(
        "--team-id", default="nextpoint",
        help="Team ID written into report metadata.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=4000,
        help="Approximate max character count per chunk (default: 4000).",
    )
    parser.add_argument(
        "--overlap", type=int, default=200,
        help="Characters of overlap between consecutive chunks (default: 200).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in args.input:
        input_path = Path(raw_path)
        if not input_path.exists():
            print(f"WARNING: file not found — {input_path}, skipping.")
            continue

        # Clean depo_id: stem with spaces/dots replaced
        depo_id = input_path.stem.replace(" ", "_").replace(".", "_")
        print(f"\nProcessing: {depo_id}")

        documents = deposition_to_documents(
            filepath=str(input_path),
            depo_id=depo_id,
            chunk_size_chars=args.chunk_size,
            overlap_chars=args.overlap,
        )

        if not documents:
            print("  Skipping — no documents created.")
            continue

        report = create_fake_report(
            documents=documents,
            query_id=depo_id,
            run_id=args.run_id,
            team_id=args.team_id,
        )

        output_file = output_dir / f"{depo_id}_report.jsonl"
        write_pydantic_json_list([report], output_file)
        print(f"  Written: {output_file}  ({len(documents)} document chunks)")

    print("\nDone.")


if __name__ == "__main__":
    main()
