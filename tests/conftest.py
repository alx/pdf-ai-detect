"""
Pytest configuration and fixtures for PDF AI detection tests.

Provides fixtures for generating test PDFs and managing output directories.
"""

import pytest
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT


@pytest.fixture(scope="session")
def test_outputs_dir():
    """
    Create and return the test outputs directory.
    This directory is used to store generated PDFs for human verification.
    """
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir


@pytest.fixture
def pdf_generator():
    """
    Provides a function to generate test PDFs with custom content.

    Returns a function that takes:
        - output_path: Path where PDF should be saved
        - content: List of text strings (each becomes a paragraph)
        - title: Optional title for the PDF
    """
    def generate_pdf(output_path: Path, content: list, title: str = "Test Document"):
        """Generate a PDF with the specified content."""
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Container for the 'Flowable' objects
        elements = []

        # Get standard styles
        styles = getSampleStyleSheet()

        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='black',
            spaceAfter=30,
            alignment=TA_LEFT,
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=12,
            leading=16,
            spaceAfter=12,
            alignment=TA_LEFT,
        )

        # Add title
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 0.2 * inch))

        # Add content paragraphs
        for text in content:
            # Clean up the text (remove extra whitespace)
            clean_text = ' '.join(text.split())
            elements.append(Paragraph(clean_text, body_style))
            elements.append(Spacer(1, 0.1 * inch))

        # Build PDF
        doc.build(elements)

        return output_path

    return generate_pdf


@pytest.fixture
def simple_human_pdf(pdf_generator, test_outputs_dir, tmp_path):
    """Generate a PDF with human-written text."""
    from tests.fixtures.sample_texts import HUMAN_TEXT_SHORT

    output_path = tmp_path / "input_human_text.pdf"
    pdf_generator(
        output_path,
        [HUMAN_TEXT_SHORT],
        title="Human-Written Text Sample"
    )
    return output_path


@pytest.fixture
def simple_ai_pdf(pdf_generator, test_outputs_dir, tmp_path):
    """Generate a PDF with AI-generated text."""
    from tests.fixtures.sample_texts import AI_TEXT_SHORT

    output_path = tmp_path / "input_ai_text.pdf"
    pdf_generator(
        output_path,
        [AI_TEXT_SHORT],
        title="AI-Generated Text Sample"
    )
    return output_path


@pytest.fixture
def mixed_content_pdf(pdf_generator, test_outputs_dir, tmp_path):
    """Generate a PDF with mixed human and AI content."""
    from tests.fixtures.sample_texts import HUMAN_TEXT_LONG, AI_TEXT_LONG, MIXED_TEXT

    output_path = tmp_path / "input_mixed_content.pdf"
    pdf_generator(
        output_path,
        [
            "Section 1: Human-Written Content",
            HUMAN_TEXT_LONG,
            "Section 2: AI-Generated Content",
            AI_TEXT_LONG,
            "Section 3: Mixed Content",
            MIXED_TEXT,
        ],
        title="Mixed Content Sample"
    )
    return output_path


@pytest.fixture
def multi_page_pdf(pdf_generator, test_outputs_dir, tmp_path):
    """Generate a multi-page PDF with varied content."""
    from tests.fixtures.sample_texts import (
        HUMAN_TEXT_SHORT,
        AI_TEXT_SHORT,
        HUMAN_TEXT_LONG,
        AI_TEXT_LONG,
    )

    output_path = tmp_path / "input_multi_page.pdf"

    # Create longer content to span multiple pages
    content = [
        "Page 1: Introduction",
        HUMAN_TEXT_SHORT,
        HUMAN_TEXT_LONG,
        "Page 2: AI-Generated Analysis",
        AI_TEXT_SHORT,
        AI_TEXT_LONG,
        "Page 3: Additional Human Commentary",
        HUMAN_TEXT_LONG,
        "Page 4: Conclusion",
        AI_TEXT_SHORT,
    ]

    pdf_generator(
        output_path,
        content,
        title="Multi-Page Test Document"
    )
    return output_path


def fetch_wikipedia_content():
    """
    Fetch random Wikipedia article content for use as human-generated text.

    Returns a tuple of (title, paragraphs_list).
    Falls back to hardcoded Wikipedia excerpt if API fails.
    """
    try:
        import requests

        # Get a random Wikipedia article
        random_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        response = requests.get(random_url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            title = data.get("title", "Wikipedia Article")
            extract = data.get("extract", "")

            # Split into paragraphs
            paragraphs = [p.strip() for p in extract.split('\n') if p.strip()]

            if paragraphs:
                return title, paragraphs
    except Exception:
        pass  # Fall back to hardcoded content

    # Fallback: Use a Wikipedia excerpt about Leonardo da Vinci
    title = "Leonardo da Vinci (Wikipedia)"
    paragraphs = [
        """Leonardo di ser Piero da Vinci (15 April 1452 – 2 May 1519) was an Italian
        polymath of the High Renaissance who was active as a painter, draughtsman, engineer,
        scientist, theorist, sculptor, and architect. While his fame initially rested on his
        achievements as a painter, he also became known for his notebooks, in which he made
        drawings and notes on a variety of subjects, including anatomy, astronomy, botany,
        cartography, painting, and paleontology.""",

        """Leonardo is widely regarded as one of the greatest painters of all time and perhaps
        the most diversely talented person ever to have lived. His innovative approaches to art
        and science influenced the development of Western art for centuries. The Mona Lisa and
        The Last Supper are among the most famous, reproduced, and parodied works of art in history.""",

        """Born out of wedlock to a successful notary and a lower-class woman in Vinci,
        Leonardo was educated in Florence by the Italian painter and sculptor Andrea del Verrocchio.
        He began his career in the city, but then spent much time in the service of Ludovico Sforza
        in Milan. Later, he worked in Florence and Milan again, as well as briefly in Rome, all
        while attracting a large following of imitators and students."""
    ]

    return title, paragraphs


@pytest.fixture
def verification_test_pdf(test_outputs_dir):
    """
    Generate a comprehensive test PDF for verifying bounding box placement.

    This fixture creates a 4-page PDF with:
    - Page 1: Wikipedia content (human-written) with varied layouts
    - Page 2: AI-generated content with structured text and table
    - Page 3: Mixed content (alternating human and AI paragraphs)
    - Page 4: Edge cases (short text, special characters, etc.)

    The generated PDF is saved to tests/outputs/input.pdf and can be used
    to verify that bounding boxes are correctly placed in the output.

    Returns: Path to the generated input.pdf
    """
    from reportlab.platypus import Table, TableStyle, ListFlowable, ListItem
    from reportlab.lib import colors
    from tests.fixtures.sample_texts import AI_TEXT_LONG, SHORT_SNIPPETS

    output_path = test_outputs_dir / "input.pdf"

    # Fetch Wikipedia content
    wiki_title, wiki_paragraphs = fetch_wikipedia_content()

    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_LEFT,
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=20,
        spaceBefore=20,
        alignment=TA_LEFT,
    )

    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#555555'),
        spaceAfter=12,
        spaceBefore=12,
        alignment=TA_LEFT,
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=12,
        leading=16,
        spaceAfter=12,
        alignment=TA_LEFT,
    )

    # ========== PAGE 1: Wikipedia Content (Human-Written) ==========
    elements.append(Paragraph("Bounding Box Verification Test PDF", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Page 1: Human-Written Content (Wikipedia)", heading_style))
    elements.append(Paragraph(
        "Expected result: <b>Green to yellow boxes</b> (low AI detection scores)",
        body_style
    ))
    elements.append(Spacer(1, 0.1 * inch))

    elements.append(Paragraph(f"<b>{wiki_title}</b>", subheading_style))

    # Add Wikipedia paragraphs
    for para in wiki_paragraphs[:3]:  # Limit to 3 paragraphs
        clean_text = ' '.join(para.split())
        elements.append(Paragraph(clean_text, body_style))

    # Add a bullet list
    elements.append(Paragraph("<b>Key characteristics of this text:</b>", subheading_style))
    bullet_items = [
        ListItem(Paragraph("Natural language with varied sentence structures", body_style)),
        ListItem(Paragraph("Historical facts and biographical information", body_style)),
        ListItem(Paragraph("Human-written Wikipedia content", body_style)),
    ]
    elements.append(ListFlowable(bullet_items, bulletType='bullet'))

    elements.append(PageBreak())

    # ========== PAGE 2: AI-Generated Content ==========
    elements.append(Paragraph("Page 2: AI-Generated Content", heading_style))
    elements.append(Paragraph(
        "Expected result: <b>Yellow to red boxes</b> (high AI detection scores)",
        body_style
    ))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Renewable Energy: A Comprehensive Analysis", subheading_style))

    # Add AI-generated text
    ai_paragraphs = [p.strip() for p in AI_TEXT_LONG.split('\n\n') if p.strip()]
    for para in ai_paragraphs[:3]:  # First 3 paragraphs
        clean_text = ' '.join(para.split())
        elements.append(Paragraph(clean_text, body_style))

    elements.append(Spacer(1, 0.2 * inch))

    # Add a simple table
    elements.append(Paragraph("<b>Comparison Table</b>", subheading_style))
    table_data = [
        ['Energy Source', 'Renewable', 'Carbon Emissions'],
        ['Solar Power', 'Yes', 'Minimal'],
        ['Wind Power', 'Yes', 'Minimal'],
        ['Coal', 'No', 'High'],
    ]

    table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)

    elements.append(PageBreak())

    # ========== PAGE 3: Mixed Content ==========
    elements.append(Paragraph("Page 3: Mixed Human and AI Content", heading_style))
    elements.append(Paragraph(
        "Expected result: <b>Varied colors</b> showing clear distinction between human and AI text",
        body_style
    ))
    elements.append(Spacer(1, 0.2 * inch))

    # Alternate between human (Wikipedia) and AI paragraphs
    mixed_sections = [
        ("[HUMAN] Personal Reflection:", wiki_paragraphs[0] if wiki_paragraphs else "Human text here."),
        ("[AI] Technical Analysis:", ai_paragraphs[0] if ai_paragraphs else AI_TEXT_LONG.split('\n\n')[0]),
        ("[HUMAN] Historical Context:", wiki_paragraphs[1] if len(wiki_paragraphs) > 1 else "More human text."),
        ("[AI] Systematic Overview:", ai_paragraphs[1] if len(ai_paragraphs) > 1 else AI_TEXT_LONG.split('\n\n')[1]),
    ]

    for label, content in mixed_sections:
        elements.append(Paragraph(f"<b>{label}</b>", subheading_style))
        clean_text = ' '.join(content.split())
        elements.append(Paragraph(clean_text, body_style))
        elements.append(Spacer(1, 0.1 * inch))

    elements.append(PageBreak())

    # ========== PAGE 4: Edge Cases ==========
    elements.append(Paragraph("Page 4: Edge Cases and Boundary Conditions", heading_style))
    elements.append(Paragraph(
        "Expected result: <b>Minimal or no coloring</b> for very short text",
        body_style
    ))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Very Short Text Snippets:</b>", subheading_style))
    for snippet in SHORT_SNIPPETS:
        elements.append(Paragraph(f"• {snippet}", body_style))

    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Special Characters and Numbers:</b>", subheading_style))
    special_cases = [
        "Email: test@example.com",
        "Phone: +1 (555) 123-4567",
        "Math: E = mc²",
        "Code: function(x) { return x * 2; }",
        "Numbers: 123,456.789",
        "Symbols: !@#$%^&*()",
        "Unicode: café, naïve, 日本語",
    ]
    for case in special_cases:
        elements.append(Paragraph(f"• {case}", body_style))

    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Single Words:</b>", subheading_style))
    elements.append(Paragraph("Technology Innovation Sustainability Development Future", body_style))

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        "<b>End of Verification Test PDF</b><br/>"
        "Process this PDF through the AI detection pipeline to verify bounding box placement.",
        body_style
    ))

    # Build the PDF
    doc.build(elements)

    return output_path
