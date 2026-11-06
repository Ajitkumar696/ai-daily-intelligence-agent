from collections import defaultdict
from html import escape
from xml.sax.saxutils import quoteattr

from ..summarization.schemas import (
    CATEGORIES,
    GENERATED_FIELDS,
    validate_summary_data
)


IMPORTANCE_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _successful_summaries(summaries):
    successful = []
    for item in summaries:
        if (
            not isinstance(item, dict)
            or item.get("summary_status") != "SUCCESS"
            or not isinstance(item.get("source_url"), str)
            or not item["source_url"].strip()
        ):
            continue

        try:
            validated = validate_summary_data({
                field: item.get(field)
                for field in GENERATED_FIELDS
            }, trusted_title=item.get("title"))
        except ValueError:
            continue

        article = dict(item)
        article.update(validated)
        article["source_url"] = item["source_url"].strip()
        successful.append(article)

    return successful


def _paragraph_text(value, fallback="Not specified in the source."):
    text = value.strip() if isinstance(value, str) else ""
    return escape(text or fallback)


def _article_flowables(article, styles):
    from reportlab.platypus import (
        HRFlowable,
        Paragraph
    )

    from .styles import RULE, importance_color

    story = []
    title = _paragraph_text(article.get("title"), "Untitled article")
    story.append(Paragraph(title, styles["ArticleHeadline"]))

    published = article.get("published_date") or "Date not provided"
    source = article.get("source") or "Unknown source"
    importance = article.get("importance") or "Not rated"
    color = importance_color(importance)
    meta = (
        f'{escape(str(source))} &nbsp;|&nbsp; {escape(str(published))}'
        f' &nbsp;|&nbsp; <font color="{color.hexval()}">'
        f'Importance: {escape(str(importance))}</font>'
    )
    story.append(Paragraph(meta, styles["SmallMeta"]))
    story.append(Paragraph(
        _paragraph_text(article.get("short_summary")),
        styles["BodyCopy"]
    ))

    points = article.get("key_points")
    if isinstance(points, list) and points:
        story.append(Paragraph("<b>Key points</b>", styles["BodyCopy"]))
        for point in points:
            story.append(Paragraph(
                _paragraph_text(point),
                styles["KeyPoint"],
                bulletText="\u2022"
            ))

    story.append(Paragraph(
        f"<b>Why it matters:</b> "
        f"{_paragraph_text(article.get('why_it_matters'))}",
        styles["BodyCopy"]
    ))

    url = str(article["source_url"])
    href = quoteattr(url)
    link_label = escape(url)
    story.append(Paragraph(
        f'<link href={href} color="#246B9E">'
        f"Open original source: {link_label}</link>",
        styles["Link"]
    ))
    story.append(HRFlowable(
        width="100%",
        thickness=0.5,
        color=RULE,
        spaceBefore=3,
        spaceAfter=7
    ))
    return story


def build_report_story(summaries, edition, report_date):
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )

    from .styles import (
        NAVY,
        PALE_BLUE,
        RULE,
        build_styles
    )

    styles = build_styles()
    successful = _successful_summaries(summaries)
    source_names = sorted({
        str(item.get("source"))
        for item in summaries
        if isinstance(item, dict) and item.get("source")
    })
    story = [
        Spacer(1, 0.8 * inch),
        Paragraph("AI Daily Intelligence", styles["CoverTitle"]),
        Paragraph(
            f"Daily technology briefing &nbsp;|&nbsp; "
            f"{escape(report_date.isoformat())} &nbsp;|&nbsp; "
            f"Edition {edition:02d}",
            styles["CoverSubtitle"]
        ),
        Spacer(1, 0.15 * inch)
    ]

    stats = [
        [Paragraph("<b>Articles processed</b>", styles["SmallMeta"]),
         Paragraph("<b>Sources</b>", styles["SmallMeta"]),
         Paragraph("<b>Verified summaries</b>", styles["SmallMeta"])],
        [str(len(summaries)), str(len(source_names)), str(len(successful))]
    ]
    metrics = Table(stats, colWidths=[2.0 * inch] * 3, hAlign="LEFT")
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 18),
        ("LEADING", (0, 0), (-1, -1), 22),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
    ]))
    story.extend([
        metrics,
        Paragraph(
            f"Executive summary ({len(successful)} verified "
            f"{'article' if len(successful) == 1 else 'articles'})",
            styles["CoverSection"]
        )
    ])

    if successful:
        summary_sentence = (
            f"{len(successful)} validated summary is included"
            if len(successful) == 1
            else f"{len(successful)} validated summaries are included"
        )
        story.append(Paragraph(
            f"{summary_sentence} in this "
            "report. The highlights below are drawn from those articles.",
            styles["BodyCopy"]
        ))
        highlights = sorted(
            successful,
            key=lambda item: (
                IMPORTANCE_ORDER.get(item.get("importance"), 3),
                str(item.get("title", "")).casefold()
            )
        )[:3]
        for article in highlights:
            title = _paragraph_text(article.get("title"))
            summary = _paragraph_text(article.get("short_summary"))
            story.append(Paragraph(
                f"<b>{title}</b><br/>{summary}",
                styles["Highlight"]
            ))
    else:
        story.append(Paragraph(
            "No successfully summarized articles are available for this "
            "edition. Failed and insufficient-content records are not "
            "presented as verified news.",
            styles["EmptyState"]
        ))

    story.append(PageBreak())

    if not successful:
        story.append(Paragraph(
            "No verified news articles",
            styles["SectionHeading"]
        ))
        story.append(Paragraph(
            "Run summarization with a configured Gemini API key and review "
            "the per-article statuses in summarized_updates.json.",
            styles["BodyCopy"]
        ))
        return story

    grouped = defaultdict(list)
    for article in successful:
        category = article.get("category")
        if category not in CATEGORIES:
            category = "Other"
        grouped[category].append(article)

    for category in CATEGORIES:
        articles = grouped.get(category)
        if not articles:
            continue
        story.append(Paragraph(escape(category), styles["SectionHeading"]))
        for article in articles:
            story.extend(_article_flowables(article, styles))

    return story
