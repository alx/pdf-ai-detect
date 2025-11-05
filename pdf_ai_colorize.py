#!/usr/bin/env python3
"""
PDF AI Detection and Colorization Tool

This script extracts text with bounding boxes from a PDF file,
uses AI detection to score each text segment, and creates a
colorized version of the PDF where colors indicate the likelihood
of text being AI-generated.

Usage:
    python pdf_ai_colorize.py input.pdf output.pdf [options]

Example:
    python pdf_ai_colorize.py document.pdf colorized.pdf --model gpt2 --opacity 0.3
"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from pdf_processor import PDFProcessor
from ai_detector import SimpleAIDetector, FastDetectGPTDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Colorize PDF bounding boxes based on AI-generated text detection scores"
    )
    parser.add_argument(
        "input_pdf",
        type=str,
        help="Path to input PDF file"
    )
    parser.add_argument(
        "output_pdf",
        type=str,
        help="Path to output colorized PDF file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model to use for AI detection (default: gpt2). Options: gpt2, gpt2-medium, gpt2-large, distilgpt2"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="simple",
        choices=["simple", "fast-detect-gpt"],
        help="Detector type to use (default: simple)"
    )
    parser.add_argument(
        "--unit-type",
        type=str,
        default="line",
        choices=["char", "word", "line"],
        help="Text unit type for extraction (default: line)"
    )
    parser.add_argument(
        "--merge-boxes",
        type=int,
        default=5,
        help="Number of text boxes to merge into segments (default: 5, set to 1 to disable merging)"
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.3,
        help="Opacity of colored boxes (0.0 to 1.0, default: 0.3)"
    )
    parser.add_argument(
        "--create-legend",
        action="store_true",
        help="Create a separate PDF with color legend"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=10,
        help="Minimum text length to analyze (default: 10 characters)"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input_pdf).exists():
        logger.error(f"Input PDF file not found: {args.input_pdf}")
        return 1

    # Validate opacity
    if not 0.0 <= args.opacity <= 1.0:
        logger.error("Opacity must be between 0.0 and 1.0")
        return 1

    try:
        # Step 1: Initialize PDF processor
        logger.info("="*60)
        logger.info("PDF AI Detection and Colorization")
        logger.info("="*60)
        logger.info(f"Input PDF: {args.input_pdf}")
        logger.info(f"Output PDF: {args.output_pdf}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Detector: {args.detector}")
        logger.info("="*60)

        processor = PDFProcessor(args.input_pdf)

        # Step 2: Extract text with bounding boxes
        logger.info("\n[1/4] Extracting text with bounding boxes...")
        boxes = processor.extract_text_with_boxes(unit_type=args.unit_type)

        if not boxes:
            logger.error("No text boxes found in PDF")
            return 1

        logger.info(f"Extracted {len(boxes)} text boxes")

        # Step 3: Merge boxes into larger segments if requested
        if args.merge_boxes > 1:
            logger.info(f"\n[2/4] Merging boxes into segments (max {args.merge_boxes} boxes per segment)...")
            boxes = processor.merge_boxes_into_segments(max_boxes_per_segment=args.merge_boxes)
            processor.bounding_boxes = boxes
            logger.info(f"Created {len(boxes)} text segments")
        else:
            logger.info("\n[2/4] Skipping box merging (merge-boxes=1)")

        # Step 4: Initialize AI detector
        logger.info(f"\n[3/4] Initializing AI detector ({args.detector})...")
        if args.detector == "simple":
            detector = SimpleAIDetector(model_name=args.model)
        else:
            detector = FastDetectGPTDetector(scoring_model_name=args.model)

        # Step 5: Score each text box
        logger.info("\n[4/4] Scoring text segments for AI detection...")
        for box in tqdm(boxes, desc="Analyzing text"):
            if len(box.text.strip()) >= args.min_text_length:
                box.score = detector.score_text(box.text)
            else:
                box.score = 0.0  # Don't score very short text

        # Step 6: Create colorized PDF
        logger.info("\nCreating colorized PDF...")
        processor.colorize_pdf(args.output_pdf, opacity=args.opacity)

        # Step 7: Create legend if requested
        if args.create_legend:
            legend_path = Path(args.output_pdf).stem + "_legend.pdf"
            logger.info(f"Creating legend PDF: {legend_path}")
            processor.create_visualization_legend(legend_path)

        # Summary statistics
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        scores = [box.score for box in boxes if box.score > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)

            logger.info(f"Total text segments analyzed: {len(scores)}")
            logger.info(f"Average AI detection score: {avg_score:.3f}")
            logger.info(f"Minimum score: {min_score:.3f}")
            logger.info(f"Maximum score: {max_score:.3f}")

            # Count segments by category
            human_like = sum(1 for s in scores if s < 0.3)
            mixed = sum(1 for s in scores if 0.3 <= s <= 0.7)
            ai_like = sum(1 for s in scores if s > 0.7)

            logger.info(f"\nSegment categories:")
            logger.info(f"  Human-like (score < 0.3): {human_like} ({100*human_like/len(scores):.1f}%)")
            logger.info(f"  Mixed (0.3 <= score <= 0.7): {mixed} ({100*mixed/len(scores):.1f}%)")
            logger.info(f"  AI-like (score > 0.7): {ai_like} ({100*ai_like/len(scores):.1f}%)")

        logger.info("="*60)
        logger.info(f"\n✓ Success! Colorized PDF saved to: {args.output_pdf}")
        logger.info("\nColor guide:")
        logger.info("  🟢 Green = Likely human-written")
        logger.info("  🟡 Yellow = Mixed/uncertain")
        logger.info("  🔴 Red = Likely AI-generated")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
