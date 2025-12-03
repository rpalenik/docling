# IBM Docling Framework - Installation and Testing Guide

This project demonstrates how to install and use IBM's Docling framework to process PDF and HTML documents using both basic and VLM (Vision-Language Model) pipelines, with JSON output format.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Installation](#installation)
- [Understanding the Pipelines](#understanding-the-pipelines)
- [Usage](#usage)
- [Output Format](#output-format)
- [Pipeline Comparison](#pipeline-comparison)

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- For VLM pipeline: GPU recommended (optional, but improves performance)

## Virtual Environment Setup

It's recommended to use a virtual environment to isolate dependencies:

### Create Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Deactivate Virtual Environment

When you're done working, you can deactivate the virtual environment:

```bash
deactivate
```

## Installation

1. **Activate your virtual environment** (if not already activated):
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `docling` - Core Docling framework
   - `docling-ibm-models` - IBM's AI models for VLM pipeline

3. **Verify installation**:
   ```bash
   python -c "import docling; print('Docling installed successfully!')"
   ```

## Understanding the Pipelines

Docling offers two main processing pipelines:

### Basic Pipeline (Default)

**Characteristics:**
- Uses traditional document parsing techniques
- Rule-based layout analysis
- Text extraction without AI models
- Fast processing
- Low resource requirements (CPU only)
- Good for well-structured documents

**Best for:**
- Standard PDF documents with clear structure
- HTML pages
- Simple layouts
- High-volume processing where speed matters
- Documents without complex visual elements

**Limitations:**
- May struggle with scanned documents
- Less accurate for complex layouts
- Limited understanding of visual context

### VLM Pipeline (Vision-Language Model)

**Characteristics:**
- Uses AI models (Granite-Docling) for document understanding
- Vision-language model integration
- Advanced layout understanding
- Better table extraction
- Improved OCR capabilities
- Slower processing
- Higher resource requirements (GPU recommended)

**Best for:**
- Scanned PDFs or image-based documents
- Complex layouts with multiple columns
- Documents with embedded images
- Tables and structured data
- Non-standard formatting
- When accuracy is more important than speed

**Limitations:**
- Slower processing time
- Requires more computational resources
- May be overkill for simple documents

## Usage

### Basic Usage

Process a PDF file with both pipelines:

```bash
python test_docling.py your_document.pdf
```

This will create two JSON output files:
- `outputs/your_document_basic.json` - Basic pipeline result
- `outputs/your_document_vlm.json` - VLM pipeline result

### Processing Both PDF and HTML

Process both PDF and HTML files:

```bash
python test_docling.py document.pdf page.html
```

This will create four JSON output files:
- `outputs/document_basic.json` - PDF with basic pipeline
- `outputs/document_vlm.json` - PDF with VLM pipeline
- `outputs/page_basic.json` - HTML with basic pipeline
- `outputs/page_vlm.json` - HTML with VLM pipeline

### Comparing Pipeline Results

Compare the outputs from both pipelines:

```bash
python compare_pipelines.py outputs/document_basic.json outputs/document_vlm.json
```

This will show:
- Structural differences between outputs
- Content analysis (text extraction comparison)
- Recommendations on when to use each pipeline

## Output Format

All outputs are saved as JSON files in the `outputs/` directory. The JSON structure contains:

- **Document metadata**: Information about the source document
- **Content structure**: Hierarchical representation of document content
- **Text content**: Extracted text from the document
- **Layout information**: Spatial information about document elements
- **Tables**: Extracted table data (if present)
- **Images**: References to embedded images (if present)

### Example JSON Structure

```json
{
  "document": {
    "metadata": {
      "source": "document.pdf",
      "format": "pdf",
      "pages": 10
    },
    "content": [
      {
        "type": "paragraph",
        "text": "Document content here...",
        "page": 1
      },
      {
        "type": "table",
        "data": [...],
        "page": 2
      }
    ]
  }
}
```

## Pipeline Comparison

### When to Use Basic Pipeline

✅ **Use Basic Pipeline when:**
- Processing many simple documents quickly
- Documents have clear, standard structure
- Speed is more important than perfect accuracy
- Working with limited computational resources
- Processing HTML pages (VLM is primarily for PDF/image documents)

### When to Use VLM Pipeline

✅ **Use VLM Pipeline when:**
- Document has complex layouts or visual elements
- Working with scanned PDFs or image-based documents
- Need accurate table extraction
- Document contains embedded images that need understanding
- Accuracy is more important than speed
- Have GPU resources available

### Performance Comparison

| Aspect | Basic Pipeline | VLM Pipeline |
|--------|---------------|--------------|
| Speed | Fast | Slower |
| Resource Usage | Low (CPU) | High (GPU recommended) |
| Accuracy (Simple Docs) | Good | Good |
| Accuracy (Complex Docs) | Moderate | Excellent |
| Scanned PDFs | Limited | Excellent |
| Tables | Basic | Advanced |
| Images | Basic extraction | Understanding |

## Project Structure

```
docling/
├── venv/                  # Virtual environment (not in git)
├── outputs/               # Output JSON files (not in git)
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── test_docling.py       # Main test script
├── compare_pipelines.py   # Pipeline comparison tool
└── README.md             # This file
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. Virtual environment is activated
2. All packages are installed: `pip install -r requirements.txt`

### VLM Pipeline Errors

If VLM pipeline fails:
- Check that `docling-ibm-models` is installed
- For Apple Silicon (M1/M2/M3), MLX acceleration is used automatically
- For other systems, ensure you have sufficient memory/GPU resources

### File Not Found Errors

- Ensure input files exist and paths are correct
- Use absolute paths if relative paths don't work
- Check file permissions

## Additional Resources

- [Docling GitHub Repository](https://github.com/DS4SD/docling)
- [IBM Granite Docling Documentation](https://www.ibm.com/granite/docs/models/docling)
- [Docling PyPI Package](https://pypi.org/project/docling/)

## Collections and Law Processing

The project includes a collections-based structure for organizing and processing Slovak law documents with their annexes.

### Collections Structure

Laws are organized in collections with YAML manifests:

```
collections/
└── {collection_name}/
    └── {law_id}/
        ├── manifest.yaml          # YAML manifest
        ├── main.html              # Main document
        ├── annexes/               # PDF annexes
        ├── cache/                 # Docling cache (per-law)
        └── output/                 # Processing outputs (per-law)
```

### Processing Laws

**Process a single law:**
```bash
python process_law.py --collection collections/dane --law 595_2003
```

**Process entire collection:**
```bash
python process_law.py --collection collections/dane
```

**Using manifest directly:**
```bash
python process_law.py --manifest collections/dane/595_2003/manifest.yaml
```

### Migrating Existing Laws

Migrate existing HTML files and data to the collections structure:

```bash
python migrate_to_collections.py \
  --html-file "input/595:2003 Z. z. - Zákon o dani z príjmov.html" \
  --collection collections/dane \
  --law-id "595/2003" \
  --annexes-dir input
```

### Sequential Parser with Annexes

The sequential parser supports annex processing:

```bash
python sequential_parser.py output/document.json \
  --manifest collections/dane/595_2003/manifest.yaml
```

For more details, see [COLLECTIONS_STRUCTURE.md](COLLECTIONS_STRUCTURE.md).

## License

This project is for testing and demonstration purposes. Please refer to IBM's documentation for licensing information about the Docling framework.

