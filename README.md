# PDF AI Detection & Colorization

Visualize AI-generated content in PDF files using colorized bounding boxes. This tool extracts text with bounding boxes from PDFs using [Docling](https://github.com/docling-project/docling), analyzes each segment with AI detection algorithms inspired by [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt), and creates a colorized PDF where:

- 🟢 **Green** = Likely human-written text
- 🟡 **Yellow** = Mixed/uncertain origin
- 🔴 **Red** = Likely AI-generated text

## Features

- **Accurate Text Extraction**: Uses Docling to extract text with precise bounding box coordinates
- **AI Detection**: Analyzes text segments using perplexity-based detection algorithms
- **Visual Output**: Creates colorized PDFs with transparent overlays indicating AI likelihood
- **Configurable**: Adjust detection models, text granularity, merging, and visualization opacity
- **Multiple Models**: Supports various language models (GPT-2, DistilGPT-2, etc.)

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pdf-ai-detect.git
cd pdf-ai-detect

# Run the setup script
bash setup.sh
```

### Manual Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# (Optional) Clone fast-detect-gpt for reference
git clone https://github.com/baoguangsheng/fast-detect-gpt.git
```

### Requirements

- Python 3.8+
- PyTorch 1.10.0+
- CUDA (optional, for GPU acceleration)
- ~2GB disk space for models

## Usage

### Basic Usage

```bash
python pdf_ai_colorize.py input.pdf output.pdf
```

### Advanced Options

```bash
python pdf_ai_colorize.py input.pdf output.pdf \
    --model gpt2-medium \
    --unit-type line \
    --merge-boxes 5 \
    --opacity 0.3 \
    --create-legend
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_pdf` | Path to input PDF file | Required |
| `output_pdf` | Path to output colorized PDF | Required |
| `--model` | Model for AI detection (gpt2, gpt2-medium, gpt2-large, distilgpt2) | gpt2 |
| `--detector` | Detector type (simple, fast-detect-gpt) | simple |
| `--unit-type` | Text extraction granularity (char, word, line) | line |
| `--merge-boxes` | Number of boxes to merge into segments (1=no merge) | 5 |
| `--opacity` | Color overlay opacity (0.0-1.0) | 0.3 |
| `--create-legend` | Generate a color scale legend PDF | False |
| `--min-text-length` | Minimum characters to analyze | 10 |

## Examples

### Example 1: Basic Analysis

```bash
python pdf_ai_colorize.py research_paper.pdf analyzed.pdf
```

### Example 2: High-Precision Analysis

Use a larger model and finer granularity for better accuracy:

```bash
python pdf_ai_colorize.py document.pdf output.pdf \
    --model gpt2-large \
    --unit-type word \
    --merge-boxes 10
```

### Example 3: Quick Preview

Use a smaller model for faster processing:

```bash
python pdf_ai_colorize.py document.pdf output.pdf \
    --model distilgpt2 \
    --merge-boxes 3
```

## How It Works

1. **Text Extraction**: Docling parses the PDF and extracts text with precise bounding box coordinates at character, word, or line level

2. **Text Segmentation**: Nearby text boxes are optionally merged into larger segments for better detection accuracy

3. **AI Detection**: Each text segment is analyzed using perplexity-based methods:
   - **Simple Detector**: Fast perplexity scoring using GPT-2 family models
   - **Fast-DetectGPT**: Advanced conditional probability curvature analysis

4. **Scoring**: Text receives a score from 0.0 (human-like) to 1.0 (AI-like) based on:
   - Perplexity (lower = more predictable = more AI-like)
   - Language model probability distributions
   - Text complexity patterns

5. **Colorization**: Bounding boxes are colorized on a green-yellow-red scale based on scores

## Technical Details

### AI Detection Methodology

This tool uses perplexity-based detection inspired by Fast-DetectGPT (Bao et al., ICLR 2024). The core idea:

- **Lower perplexity** → More predictable text → Likely AI-generated
- **Higher perplexity** → Less predictable text → Likely human-written

The detector calculates log-likelihood scores using pretrained language models and converts them to probability estimates.

### Supported Models

- **distilgpt2**: Fast, lightweight (~300MB)
- **gpt2**: Balanced speed/accuracy (~500MB)
- **gpt2-medium**: Better accuracy (~1.5GB)
- **gpt2-large**: Best accuracy (~3GB)

### Performance

- **Processing Speed**: ~1-5 pages/minute (depending on model and hardware)
- **GPU Acceleration**: Automatically used if available (5-10x faster)
- **Memory Usage**: 2-8GB RAM (depending on model)

## Project Structure

```
pdf-ai-detect/
├── pdf_ai_colorize.py    # Main script
├── pdf_processor.py      # PDF text extraction and colorization
├── ai_detector.py        # AI detection algorithms
├── requirements.txt      # Python dependencies
├── setup.sh             # Setup script
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Limitations

- **Accuracy**: Detection is probabilistic and not 100% accurate
- **Short Text**: Very short segments (<10 words) are unreliable
- **Language**: Currently optimized for English text
- **PDF Types**: Works best with text-based PDFs (not scanned images)
- **Model Bias**: Detection quality depends on the model used

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Resources

- [Docling Documentation](https://docling-project.github.io/docling/)
- [Fast-DetectGPT Paper](https://arxiv.org/abs/2310.05130)
- [Fast-DetectGPT Repository](https://github.com/baoguangsheng/fast-detect-gpt)

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in research, please cite the Fast-DetectGPT paper:

```bibtex
@inproceedings{bao2024fast,
  title={Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature},
  author={Bao, Guangsheng and Zhao, Yanbin and Teng, Zhiyang and Yang, Linyi and Zhang, Yue},
  booktitle={ICLR},
  year={2024}
}
```

## Acknowledgments

- [Docling Project](https://github.com/docling-project) for excellent PDF parsing tools
- [Bao et al.](https://github.com/baoguangsheng/fast-detect-gpt) for Fast-DetectGPT algorithm
- [Hugging Face](https://huggingface.co/) for transformer models
