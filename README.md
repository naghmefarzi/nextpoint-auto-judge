# NextPoint Deposition Annotation

Annotate NextPoint deposition transcripts using the TREC AutoJudge annotation interface.

## Installation

```bash
# Install uv if you don't have it
pip install uv

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Running the Annotation Interface

Generate the annotation HTML file:

```bash
autojudge-annotate generate \
    --rag-responses results \
    --rag-topics topics/nextpoint-deposition/nextpoint-deposition.jsonl \
    --output nextpoint-annotator.html \
    --dataset nextpoint \
    --show-documents
```

Then open `nextpoint-annotator.html` in your browser.

### Annotating Documents

1. Open the generated HTML file in your browser.
2. Select **Documents** from the mode selector in the top bar.
3. Browse and annotate individual deposition chunks per topic.
