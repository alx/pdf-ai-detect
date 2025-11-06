"""
Integration tests for PDF AI detection and colorization.

These tests generate PDFs with known content, run the full detection pipeline,
and save the colorized outputs for human verification in tests/outputs/.

To run these tests:
    pytest tests/test_integration.py -v

Test outputs will be saved to: tests/outputs/
"""

import pytest
from pathlib import Path
from datetime import datetime
import shutil

from pdf_processor import PDFProcessor
from ai_detector import SimpleAIDetector, FastDetectGPTDetector


class TestSimpleDetector:
    """Integration tests using SimpleAIDetector (faster, perplexity-based)."""

    def test_human_text_detection(self, simple_human_pdf, test_outputs_dir):
        """
        Test detection of human-written text with SimpleAIDetector.

        Expected: Most text should have low scores (green/yellow),
        indicating human-like characteristics.

        Output saved to: tests/outputs/test_human_simple_TIMESTAMP.pdf
        """
        # Initialize processor
        processor = PDFProcessor(str(simple_human_pdf))

        # Extract text with bounding boxes
        boxes = processor.extract_text_with_boxes(unit_type="line")
        assert len(boxes) > 0, "No text boxes extracted from PDF"

        # Initialize detector
        detector = SimpleAIDetector(model_name="gpt2")

        # Score each text box
        for box in boxes:
            if len(box.text.strip()) >= 10:
                box.score = detector.score_text(box.text)
            else:
                box.score = 0.0

        # Create colorized output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"test_human_simple_{timestamp}.pdf"
        output_path = test_outputs_dir / output_filename

        processor.colorize_pdf(str(output_path), opacity=0.3)

        # Verify output was created
        assert output_path.exists(), "Colorized PDF was not created"
        assert output_path.stat().st_size > 0, "Colorized PDF is empty"

        # Calculate statistics for logging
        scores = [box.score for box in boxes if box.score > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\n[Human Text - SimpleDetector]")
            print(f"  Average score: {avg_score:.3f}")
            print(f"  Min score: {min(scores):.3f}")
            print(f"  Max score: {max(scores):.3f}")
            print(f"  Output: {output_path}")

            # Human text should generally have lower scores (less AI-like)
            # This is a soft assertion - may vary with different texts
            assert avg_score < 0.8, "Human text scored too high (AI-like)"

    def test_ai_text_detection(self, simple_ai_pdf, test_outputs_dir):
        """
        Test detection of AI-generated text with SimpleAIDetector.

        Expected: Text should have higher scores (yellow/red),
        indicating AI-generated characteristics.

        Output saved to: tests/outputs/test_ai_simple_TIMESTAMP.pdf
        """
        # Initialize processor
        processor = PDFProcessor(str(simple_ai_pdf))

        # Extract text with bounding boxes
        boxes = processor.extract_text_with_boxes(unit_type="line")
        assert len(boxes) > 0, "No text boxes extracted from PDF"

        # Initialize detector
        detector = SimpleAIDetector(model_name="gpt2")

        # Score each text box
        for box in boxes:
            if len(box.text.strip()) >= 10:
                box.score = detector.score_text(box.text)
            else:
                box.score = 0.0

        # Create colorized output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"test_ai_simple_{timestamp}.pdf"
        output_path = test_outputs_dir / output_filename

        processor.colorize_pdf(str(output_path), opacity=0.3)

        # Verify output was created
        assert output_path.exists(), "Colorized PDF was not created"
        assert output_path.stat().st_size > 0, "Colorized PDF is empty"

        # Calculate statistics for logging
        scores = [box.score for box in boxes if box.score > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\n[AI Text - SimpleDetector]")
            print(f"  Average score: {avg_score:.3f}")
            print(f"  Min score: {min(scores):.3f}")
            print(f"  Max score: {max(scores):.3f}")
            print(f"  Output: {output_path}")

            # AI text should generally have higher scores
            # This is a soft assertion - may vary with different texts
            assert avg_score > 0.3, "AI text scored too low (human-like)"


class TestFastDetectGPT:
    """Integration tests using FastDetectGPTDetector (slower, more accurate)."""

    def test_mixed_content_detection(self, mixed_content_pdf, test_outputs_dir):
        """
        Test detection of mixed human/AI content with FastDetectGPTDetector.

        Expected: Should show variation in scores across different sections,
        with human sections scoring lower and AI sections scoring higher.

        Output saved to: tests/outputs/test_mixed_fastdetect_TIMESTAMP.pdf

        Note: This test may take longer due to FastDetectGPT's complexity.
        """
        # Initialize processor
        processor = PDFProcessor(str(mixed_content_pdf))

        # Extract text with bounding boxes (use larger segments)
        boxes = processor.extract_text_with_boxes(unit_type="line")
        assert len(boxes) > 0, "No text boxes extracted from PDF"

        # Merge boxes for better detection accuracy
        boxes = processor.merge_boxes_into_segments(max_boxes_per_segment=5)
        processor.bounding_boxes = boxes

        # Initialize detector
        detector = FastDetectGPTDetector(scoring_model_name="gpt2")

        # Score each text segment
        for box in boxes:
            if len(box.text.strip()) >= 20:
                box.score = detector.score_text(box.text)
            else:
                box.score = 0.0

        # Create colorized output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"test_mixed_fastdetect_{timestamp}.pdf"
        output_path = test_outputs_dir / output_filename

        processor.colorize_pdf(str(output_path), opacity=0.3)

        # Also create a legend
        legend_filename = f"test_mixed_fastdetect_{timestamp}_legend.pdf"
        legend_path = test_outputs_dir / legend_filename
        processor.create_visualization_legend(str(legend_path))

        # Verify outputs were created
        assert output_path.exists(), "Colorized PDF was not created"
        assert output_path.stat().st_size > 0, "Colorized PDF is empty"
        assert legend_path.exists(), "Legend PDF was not created"

        # Calculate statistics
        scores = [box.score for box in boxes if box.score > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\n[Mixed Content - FastDetectGPT]")
            print(f"  Average score: {avg_score:.3f}")
            print(f"  Min score: {min(scores):.3f}")
            print(f"  Max score: {max(scores):.3f}")
            print(f"  Score variance: {max(scores) - min(scores):.3f}")
            print(f"  Output: {output_path}")
            print(f"  Legend: {legend_path}")

            # Mixed content should show score variation
            score_range = max(scores) - min(scores)
            assert score_range > 0.05, "Scores show insufficient variation for mixed content"

    def test_multi_page_document(self, multi_page_pdf, test_outputs_dir):
        """
        Test detection on multi-page document with FastDetectGPTDetector.

        Expected: Should successfully process all pages and create
        colorized overlays across the entire document.

        Output saved to: tests/outputs/test_multipage_fastdetect_TIMESTAMP.pdf

        Note: This test may take longer due to document length.
        """
        # Initialize processor
        processor = PDFProcessor(str(multi_page_pdf))

        # Extract text with bounding boxes
        boxes = processor.extract_text_with_boxes(unit_type="line")
        assert len(boxes) > 0, "No text boxes extracted from PDF"

        # Verify multi-page extraction
        page_numbers = set(box.page_no for box in boxes)
        assert len(page_numbers) > 1, "PDF should have multiple pages"

        # Merge boxes for better detection
        boxes = processor.merge_boxes_into_segments(max_boxes_per_segment=3)
        processor.bounding_boxes = boxes

        # Initialize detector
        detector = FastDetectGPTDetector(scoring_model_name="gpt2")

        # Score each text segment
        for box in boxes:
            if len(box.text.strip()) >= 20:
                box.score = detector.score_text(box.text)
            else:
                box.score = 0.0

        # Create colorized output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"test_multipage_fastdetect_{timestamp}.pdf"
        output_path = test_outputs_dir / output_filename

        processor.colorize_pdf(str(output_path), opacity=0.3)

        # Verify output was created
        assert output_path.exists(), "Colorized PDF was not created"
        assert output_path.stat().st_size > 0, "Colorized PDF is empty"

        # Calculate statistics per page
        scores = [box.score for box in boxes if box.score > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\n[Multi-Page Document - FastDetectGPT]")
            print(f"  Pages processed: {len(page_numbers)}")
            print(f"  Total segments: {len(boxes)}")
            print(f"  Average score: {avg_score:.3f}")
            print(f"  Min score: {min(scores):.3f}")
            print(f"  Max score: {max(scores):.3f}")
            print(f"  Output: {output_path}")


class TestBasicPipeline:
    """Basic sanity tests to verify the pipeline works end-to-end."""

    def test_pipeline_with_simple_detector(self, simple_human_pdf, test_outputs_dir):
        """
        Basic test to verify the pipeline runs successfully with SimpleAIDetector.

        This is a quick smoke test to ensure all components work together.
        """
        processor = PDFProcessor(str(simple_human_pdf))
        boxes = processor.extract_text_with_boxes(unit_type="line")

        detector = SimpleAIDetector(model_name="gpt2")

        for box in boxes:
            if len(box.text.strip()) >= 10:
                box.score = detector.score_text(box.text)
            else:
                box.score = 0.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = test_outputs_dir / f"test_pipeline_basic_{timestamp}.pdf"

        processor.colorize_pdf(str(output_path), opacity=0.3)

        assert output_path.exists()
        print(f"\n[Basic Pipeline Test]")
        print(f"  Status: PASSED")
        print(f"  Output: {output_path}")

    def test_legend_generation(self, simple_human_pdf, test_outputs_dir):
        """
        Test that legend generation works correctly.
        """
        processor = PDFProcessor(str(simple_human_pdf))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        legend_path = test_outputs_dir / f"test_legend_{timestamp}.pdf"

        processor.create_visualization_legend(str(legend_path))

        assert legend_path.exists()
        assert legend_path.stat().st_size > 0
        print(f"\n[Legend Generation Test]")
        print(f"  Status: PASSED")
        print(f"  Output: {legend_path}")


class TestBoundingBoxVerification:
    """Test suite for verifying bounding box placement using comprehensive test PDFs."""

    def test_generate_verification_pdf(self, verification_test_pdf, test_outputs_dir):
        """
        Generate comprehensive input.pdf and process it to create output.pdf
        for manual verification of bounding box placement.

        This test creates:
        - tests/outputs/input.pdf: Comprehensive test PDF with varied content
        - tests/outputs/output.pdf: Colorized version with bounding boxes

        The test PDF includes:
        - Page 1: Wikipedia content (human-written) with varied layouts
        - Page 2: AI-generated content with structured text and table
        - Page 3: Mixed content (alternating human and AI paragraphs)
        - Page 4: Edge cases (short text, special characters, etc.)

        Expected results:
        - Page 1: Mostly green/yellow boxes (low AI scores)
        - Page 2: Mostly yellow/red boxes (high AI scores)
        - Page 3: Mixed colors showing clear distinction
        - Page 4: Minimal coloring for very short text

        To verify: Open both PDFs side-by-side and check that colored
        bounding boxes in output.pdf align correctly with text in input.pdf.
        """
        # The fixture has already generated input.pdf
        input_pdf = verification_test_pdf
        assert input_pdf.exists(), "Input PDF was not created"
        assert input_pdf.stat().st_size > 0, "Input PDF is empty"

        print(f"\n[Bounding Box Verification Test]")
        print(f"  Input PDF: {input_pdf}")

        # Initialize processor
        processor = PDFProcessor(str(input_pdf))

        # Extract text with bounding boxes at line level
        boxes = processor.extract_text_with_boxes(unit_type="line")
        assert len(boxes) > 0, "No text boxes extracted from PDF"

        # Merge boxes into larger segments for better AI detection accuracy
        boxes = processor.merge_boxes_into_segments(max_boxes_per_segment=5)
        processor.bounding_boxes = boxes

        print(f"  Extracted {len(boxes)} text segments")

        # Initialize FastDetectGPT detector for more accurate scoring
        detector = FastDetectGPTDetector(scoring_model_name="gpt2")

        # Score each text segment
        min_text_length = 20
        scored_count = 0

        for box in boxes:
            text_length = len(box.text.strip())
            if text_length >= min_text_length:
                box.score = detector.score_text(box.text)
                scored_count += 1
            else:
                box.score = 0.0

        print(f"  Scored {scored_count} segments (minimum length: {min_text_length} chars)")

        # Create colorized output.pdf
        output_path = test_outputs_dir / "output.pdf"
        processor.colorize_pdf(str(output_path), opacity=0.3)

        # Also create a legend for reference
        legend_path = test_outputs_dir / "legend.pdf"
        processor.create_visualization_legend(str(legend_path))

        # Verify outputs were created
        assert output_path.exists(), "Colorized PDF was not created"
        assert output_path.stat().st_size > 0, "Colorized PDF is empty"
        assert legend_path.exists(), "Legend PDF was not created"

        # Calculate and display statistics
        scores = [box.score for box in boxes if box.score > 0]

        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\n  Statistics:")
            print(f"    Average score: {avg_score:.3f}")
            print(f"    Min score: {min(scores):.3f}")
            print(f"    Max score: {max(scores):.3f}")
            print(f"    Score range: {max(scores) - min(scores):.3f}")

        # Count scores by category
        human_scores = [s for s in scores if s < 0.4]
        mixed_scores = [s for s in scores if 0.4 <= s < 0.6]
        ai_scores = [s for s in scores if s >= 0.6]

        print(f"\n  Score Distribution:")
        print(f"    Human-like (< 0.4): {len(human_scores)} segments")
        print(f"    Mixed (0.4-0.6): {len(mixed_scores)} segments")
        print(f"    AI-like (>= 0.6): {len(ai_scores)} segments")

        print(f"\n  Outputs:")
        print(f"    Input:  {input_pdf}")
        print(f"    Output: {output_path}")
        print(f"    Legend: {legend_path}")

        print(f"\n  Verification Instructions:")
        print(f"    1. Open both input.pdf and output.pdf")
        print(f"    2. Compare them side-by-side")
        print(f"    3. Verify that colored boxes align with text")
        print(f"    4. Check that colors match expected patterns:")
        print(f"       - Page 1 (Wikipedia): Green/Yellow boxes")
        print(f"       - Page 2 (AI content): Yellow/Red boxes")
        print(f"       - Page 3 (Mixed): Varied colors")
        print(f"       - Page 4 (Edge cases): Minimal coloring")

        # Assert that we got a reasonable score distribution
        # Note: The detector may score all text similarly depending on content
        # The main purpose is to verify bounding box placement, not scoring accuracy
        assert len(scores) > 0, "No text was scored"

        # Optional: Check for score variation (may fail with some detectors)
        score_variation = max(scores) - min(scores)
        if score_variation < 0.1:
            print(f"\n  Note: Limited score variation ({score_variation:.3f})")
            print(f"        This is expected with some content/detector combinations")
            print(f"        The bounding boxes are still valid for verification")
