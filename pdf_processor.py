"""
PDF Processing Module for extracting text with bounding boxes
and colorizing based on AI detection scores.
"""

import fitz  # PyMuPDF
from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument
from docling_core.types.doc.page import TextCellUnit
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoundingBox:
    """Represents a bounding box with text and coordinates."""

    def __init__(self, text: str, rect: Tuple[float, float, float, float], page_no: int):
        """
        Initialize bounding box.

        Args:
            text: Text content
            rect: Rectangle coordinates (x0, y0, x1, y1)
            page_no: Page number (0-indexed)
        """
        self.text = text
        self.rect = rect  # (x0, y0, x1, y1)
        self.page_no = page_no
        self.score = 0.0  # AI detection score (0.0 to 1.0)

    def __repr__(self):
        return f"BoundingBox(page={self.page_no}, rect={self.rect}, text='{self.text[:30]}...', score={self.score:.2f})"


class PDFProcessor:
    """Process PDF files to extract text with bounding boxes and colorize them."""

    def __init__(self, pdf_path: str):
        """
        Initialize PDF processor.

        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.bounding_boxes: List[BoundingBox] = []

    def extract_text_with_boxes(self, unit_type: str = "line") -> List[BoundingBox]:
        """
        Extract text with bounding boxes from PDF using docling.

        Args:
            unit_type: Type of text unit to extract ("char", "word", or "line")

        Returns:
            List of BoundingBox objects
        """
        logger.info(f"Extracting text with bounding boxes from {self.pdf_path}")

        # Map string to TextCellUnit enum
        unit_type_map = {
            "char": TextCellUnit.CHAR,
            "word": TextCellUnit.WORD,
            "line": TextCellUnit.LINE,
        }

        if unit_type not in unit_type_map:
            raise ValueError(f"Invalid unit_type: {unit_type}. Must be one of {list(unit_type_map.keys())}")

        unit = unit_type_map[unit_type]

        # Parse PDF with docling
        parser = DoclingPdfParser()
        pdf_doc: PdfDocument = parser.load(path_or_stream=str(self.pdf_path))

        self.bounding_boxes = []

        # Iterate over pages and extract text cells
        for page_no, page in pdf_doc.iterate_pages():
            logger.info(f"Processing page {page_no}")

            for cell in page.iterate_cells(unit_type=unit):
                # cell.rect is a Rectangle object with bbox property
                # bbox is (x0, y0, x1, y1)
                bbox = cell.rect.bbox if hasattr(cell.rect, 'bbox') else (
                    cell.rect.x, cell.rect.y,
                    cell.rect.x + cell.rect.width,
                    cell.rect.y + cell.rect.height
                )

                box = BoundingBox(
                    text=cell.text,
                    rect=bbox,
                    page_no=page_no
                )
                self.bounding_boxes.append(box)

        logger.info(f"Extracted {len(self.bounding_boxes)} text boxes")
        return self.bounding_boxes

    def merge_boxes_into_segments(self, max_boxes_per_segment: int = 10) -> List[BoundingBox]:
        """
        Merge nearby bounding boxes into larger text segments for better AI detection.
        This is useful because AI detectors work better on longer text.

        Args:
            max_boxes_per_segment: Maximum number of boxes to merge into one segment

        Returns:
            List of merged BoundingBox objects
        """
        if not self.bounding_boxes:
            return []

        merged_boxes = []
        current_page = self.bounding_boxes[0].page_no
        current_segment_boxes = []

        for box in self.bounding_boxes:
            # Start a new segment if page changes or we have enough boxes
            if box.page_no != current_page or len(current_segment_boxes) >= max_boxes_per_segment:
                if current_segment_boxes:
                    # Merge current segment
                    merged_box = self._merge_boxes(current_segment_boxes)
                    merged_boxes.append(merged_box)

                current_segment_boxes = [box]
                current_page = box.page_no
            else:
                current_segment_boxes.append(box)

        # Don't forget the last segment
        if current_segment_boxes:
            merged_box = self._merge_boxes(current_segment_boxes)
            merged_boxes.append(merged_box)

        logger.info(f"Merged {len(self.bounding_boxes)} boxes into {len(merged_boxes)} segments")
        return merged_boxes

    def _merge_boxes(self, boxes: List[BoundingBox]) -> BoundingBox:
        """Merge multiple boxes into one."""
        if not boxes:
            raise ValueError("Cannot merge empty list of boxes")

        # Combine text with spaces
        combined_text = " ".join(box.text for box in boxes)

        # Calculate bounding rectangle that encompasses all boxes
        x0 = min(box.rect[0] for box in boxes)
        y0 = min(box.rect[1] for box in boxes)
        x1 = max(box.rect[2] for box in boxes)
        y1 = max(box.rect[3] for box in boxes)

        return BoundingBox(
            text=combined_text,
            rect=(x0, y0, x1, y1),
            page_no=boxes[0].page_no
        )

    def score_to_color(self, score: float) -> Tuple[float, float, float]:
        """
        Convert AI detection score to RGB color.
        Low score (human-written) = green
        High score (AI-generated) = red
        Medium score = yellow/orange

        Args:
            score: Score between 0.0 and 1.0

        Returns:
            RGB tuple with values between 0.0 and 1.0
        """
        # Interpolate between green (0.0) and red (1.0) through yellow
        if score < 0.5:
            # Green to yellow
            r = score * 2
            g = 1.0
            b = 0.0
        else:
            # Yellow to red
            r = 1.0
            g = 2.0 * (1.0 - score)
            b = 0.0

        return (r, g, b)

    def colorize_pdf(self, output_path: str, opacity: float = 0.3) -> None:
        """
        Create a new PDF with colorized bounding boxes based on AI detection scores.

        Args:
            output_path: Path for the output PDF
            opacity: Opacity of the colored boxes (0.0 to 1.0)
        """
        logger.info(f"Colorizing PDF and saving to {output_path}")

        # Open PDF with PyMuPDF
        doc = fitz.open(self.pdf_path)

        # Draw colored rectangles for each bounding box
        for box in self.bounding_boxes:
            if box.page_no >= len(doc):
                logger.warning(f"Page {box.page_no} out of range, skipping")
                continue

            page = doc[box.page_no]

            # Get color based on score
            color = self.score_to_color(box.score)

            # Create rectangle
            rect = fitz.Rect(box.rect)

            # Draw filled rectangle with transparency
            page.draw_rect(
                rect,
                color=color,
                fill=color,
                overlay=True,
                fill_opacity=opacity,
            )

        # Save the modified PDF
        doc.save(output_path)
        doc.close()

        logger.info(f"Saved colorized PDF to {output_path}")

    def create_visualization_legend(self, output_path: str) -> None:
        """
        Create a simple legend page showing the color scale.

        Args:
            output_path: Path for the legend PDF
        """
        # Create a new PDF with a legend
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)

        # Title
        page.insert_text(
            (20, 30),
            "AI Detection Score Legend",
            fontsize=14,
            fontname="helv",
            color=(0, 0, 0),
        )

        # Draw color gradient
        y_pos = 50
        for i in range(11):
            score = i / 10.0
            color = self.score_to_color(score)
            rect = fitz.Rect(20, y_pos + i * 10, 100, y_pos + (i + 1) * 10)
            page.draw_rect(rect, color=color, fill=color)

            # Add label
            label = f"{score:.1f} - {'AI-generated' if score > 0.7 else 'Human' if score < 0.3 else 'Mixed'}"
            page.insert_text(
                (110, y_pos + i * 10 + 8),
                label,
                fontsize=9,
                fontname="helv",
                color=(0, 0, 0),
            )

        doc.save(output_path)
        doc.close()
        logger.info(f"Saved legend to {output_path}")
