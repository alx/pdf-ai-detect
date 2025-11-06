# PDF AI Detection Tests

This directory contains integration tests for the PDF AI detection and colorization system. The tests generate sample PDFs with known content, run them through the detection pipeline, and save colorized outputs for human verification.

## Setup

1. **Install testing dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

   This will install:
   - `pytest` - Testing framework
   - `pytest-cov` - Code coverage reporting
   - `reportlab` - PDF generation for test fixtures

2. **Ensure the main dependencies are installed:**
   The tests require the same dependencies as the main application (docling, PyMuPDF, torch, transformers, etc.). If you can run `pdf_ai_colorize.py`, you're all set.

## Running Tests

### Run all tests:
```bash
pytest tests/ -v
```

### Run specific test classes:
```bash
# SimpleAIDetector tests (faster)
pytest tests/test_integration.py::TestSimpleDetector -v

# FastDetectGPT tests (slower, more accurate)
pytest tests/test_integration.py::TestFastDetectGPT -v

# Basic pipeline tests
pytest tests/test_integration.py::TestBasicPipeline -v
```

### Run a specific test:
```bash
pytest tests/test_integration.py::TestSimpleDetector::test_human_text_detection -v
```

### Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html to view coverage report
```

## Test Structure

### Test Classes

1. **TestSimpleDetector** - Uses `SimpleAIDetector` (perplexity-based, faster)
   - `test_human_text_detection()` - Tests human-written text
   - `test_ai_text_detection()` - Tests AI-generated text

2. **TestFastDetectGPT** - Uses `FastDetectGPTDetector` (more accurate, slower)
   - `test_mixed_content_detection()` - Tests mixed human/AI content
   - `test_multi_page_document()` - Tests multi-page document processing

3. **TestBasicPipeline** - Quick smoke tests
   - `test_pipeline_with_simple_detector()` - Basic end-to-end test
   - `test_legend_generation()` - Tests legend creation

### Test Fixtures

Located in `tests/fixtures/sample_texts.py`:
- `HUMAN_TEXT_SHORT/LONG` - Human-written content (varied, natural)
- `AI_TEXT_SHORT/LONG` - AI-generated content (structured, predictable)
- `MIXED_TEXT` - Combination of human and AI characteristics

## Output Files

All test outputs are saved to `tests/outputs/` with descriptive, timestamped filenames:

```
tests/outputs/
├── test_human_simple_20251105_143022.pdf          # Human text with SimpleAIDetector
├── test_ai_simple_20251105_143045.pdf             # AI text with SimpleAIDetector
├── test_mixed_fastdetect_20251105_143112.pdf      # Mixed content with FastDetectGPT
├── test_mixed_fastdetect_20251105_143112_legend.pdf
├── test_multipage_fastdetect_20251105_143156.pdf  # Multi-page document
├── test_pipeline_basic_20251105_143201.pdf        # Basic pipeline test
└── test_legend_20251105_143205.pdf                # Legend generation test
```

**Note:** Test output PDFs are gitignored to avoid repository bloat.

## Interpreting Results

### Color Coding

The colorized PDFs use transparent overlays to indicate AI detection scores:

- **🟢 Green (score 0.0 - 0.3):** Likely human-written
  - Higher perplexity (less predictable)
  - More varied language patterns
  - Natural imperfections

- **🟡 Yellow (score 0.3 - 0.7):** Mixed/uncertain
  - Moderate perplexity
  - Could be either human or AI
  - Transitional content

- **🔴 Red (score 0.7 - 1.0):** Likely AI-generated
  - Lower perplexity (more predictable)
  - Consistent, structured patterns
  - Formal, uniform language

### Expected Test Behavior

1. **Human Text Tests:**
   - Should show mostly green/yellow colors
   - Average scores typically < 0.6
   - More variation in scores

2. **AI Text Tests:**
   - Should show mostly yellow/red colors
   - Average scores typically > 0.5
   - More consistent scores

3. **Mixed Content Tests:**
   - Should show varied colors across different sections
   - Score range (max - min) should be > 0.2
   - Clear visual distinction between sections

4. **Multi-Page Tests:**
   - All pages should be processed
   - Colors should appear consistently across pages
   - File size should be reasonable (not bloated)

### Console Output

Each test prints statistics to help with verification:

```
[Human Text - SimpleDetector]
  Average score: 0.423
  Min score: 0.156
  Max score: 0.687
  Output: tests/outputs/test_human_simple_20251105_143022.pdf

[AI Text - SimpleDetector]
  Average score: 0.645
  Min score: 0.512
  Max score: 0.789
  Output: tests/outputs/test_ai_simple_20251105_143045.pdf
```

## Manual Verification

After running tests, manually review the output PDFs:

1. **Open the colorized PDFs** in a PDF viewer (Adobe Acrobat, Preview, etc.)

2. **Check the color overlays:**
   - Are they visible but not too opaque?
   - Do they align with the text bounding boxes?
   - Do colors make sense for the content?

3. **Compare with expectations:**
   - Human text → More green
   - AI text → More red
   - Mixed content → Varied colors

4. **Check for issues:**
   - Missing overlays
   - Misaligned bounding boxes
   - Incorrect page rendering
   - Text extraction problems

## Troubleshooting

### Tests fail with "No module named 'reportlab'"
```bash
uv pip install reportlab
```

### Tests are very slow
- FastDetectGPT tests are computationally expensive
- Run SimpleDetector tests for quick validation
- Consider using smaller test documents

### Output PDFs are blank or corrupted
- Check that input PDFs were generated correctly
- Verify text extraction is working: `processor.extract_text_with_boxes()`
- Ensure models are loading properly

### Scores seem incorrect
- Remember that detection is probabilistic, not deterministic
- Scores may vary based on:
  - Text length (shorter text is harder to classify)
  - Model choice (gpt2 vs gpt2-medium, etc.)
  - Text type (technical vs conversational, etc.)

### "CUDA out of memory" errors
- Use CPU instead: Set environment variable `CUDA_VISIBLE_DEVICES=""`
- Or use smaller models: `model_name="distilgpt2"`

## Adding New Tests

To add new test cases:

1. **Add new sample text** to `fixtures/sample_texts.py`
2. **Create new fixture** in `conftest.py` to generate the PDF
3. **Write new test** in `test_integration.py`:

```python
def test_my_new_case(self, my_new_pdf_fixture, test_outputs_dir):
    """Test description."""
    processor = PDFProcessor(str(my_new_pdf_fixture))
    boxes = processor.extract_text_with_boxes(unit_type="line")

    detector = SimpleAIDetector(model_name="gpt2")
    for box in boxes:
        if len(box.text.strip()) >= 10:
            box.score = detector.score_text(box.text)

    output_path = test_outputs_dir / "test_my_case.pdf"
    processor.colorize_pdf(str(output_path), opacity=0.3)

    assert output_path.exists()
```

## CI/CD Integration

To run tests in CI/CD pipelines:

```bash
# Install dependencies (including test deps)
uv pip install -r requirements.txt

# Run tests with output directory creation
mkdir -p tests/outputs
pytest tests/ -v --tb=short

# Optionally upload test outputs as artifacts
# (depends on your CI system: GitHub Actions, GitLab CI, etc.)
```

## Notes

- **Model downloads:** First run will download models (GPT-2, ~500MB). Subsequent runs use cached models.
- **Test duration:** SimpleDetector tests take ~30s, FastDetectGPT tests take ~2-5 minutes.
- **PDF generation:** Test PDFs are generated programmatically using reportlab.
- **Reproducibility:** Tests use fixed sample texts but scores may vary slightly due to model randomness.

## Questions or Issues?

If you encounter problems:
1. Check that all dependencies are installed correctly
2. Review the console output for error messages
3. Verify that the main application works: `python pdf_ai_colorize.py --help`
4. Check the test output PDFs manually for visual debugging
