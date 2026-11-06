from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch


NAVY = colors.HexColor("#10243A")
BLUE = colors.HexColor("#246B9E")
PALE_BLUE = colors.HexColor("#EAF3F8")
INK = colors.HexColor("#243746")
MUTED = colors.HexColor("#647786")
RULE = colors.HexColor("#D6E0E7")
GREEN = colors.HexColor("#237A57")
AMBER = colors.HexColor("#9A6500")
RED = colors.HexColor("#A83B3B")


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=18,
        textColor=MUTED,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=12,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name="ArticleHeadline",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
        splitLongWords=1
    ))
    styles.add(ParagraphStyle(
        name="BodyCopy",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=INK,
        spaceAfter=6,
        splitLongWords=1
    ))
    styles.add(ParagraphStyle(
        name="KeyPoint",
        parent=styles["BodyCopy"],
        leftIndent=18,
        bulletFontName="Helvetica",
        bulletFontSize=7,
        bulletDedent=6
    ))
    styles.add(ParagraphStyle(
        name="SmallMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=MUTED,
        spaceAfter=5,
        splitLongWords=1
    ))
    styles.add(ParagraphStyle(
        name="CoverSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=BLUE,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name="Highlight",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        leftIndent=12,
        borderColor=RULE,
        borderWidth=0,
        textColor=INK,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="EmptyState",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        textColor=MUTED,
        backColor=PALE_BLUE,
        borderColor=RULE,
        borderWidth=0.5,
        borderPadding=12,
        spaceBefore=12,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="Link",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=BLUE,
        spaceAfter=10,
        splitLongWords=1
    ))
    styles["Normal"].alignment = TA_LEFT
    return styles


def importance_color(importance):
    return {
        "High": RED,
        "Medium": AMBER,
        "Low": GREEN
    }.get(importance, MUTED)
