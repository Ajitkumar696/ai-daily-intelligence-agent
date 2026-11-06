from html import escape
from urllib.parse import urlsplit
from xml.sax.saxutils import quoteattr

from ..research.schemas import validate_research_output
from .styles import build_styles


_BRIEF_SECTIONS = (
    ("Background and context", "background"),
    ("What happened", "what_happened"),
    ("Technology explained", "technology_explanation"),
    ("Important developments", "important_developments"),
    ("Technical details", "technical_details"),
    ("Companies, models and technologies", "companies_models_technologies"),
    ("Real-world applications", "real_world_applications"),
    ("Benefits", "benefits"),
    ("Limitations", "limitations"),
    ("Potential impact", "potential_impact"),
    ("Why this matters", "why_it_matters")
)


def _http_url(value):
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlsplit(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def successful_research_briefs(records):
    successful = []
    for item in records:
        if not isinstance(item, dict) or item.get("research_status") != "SUCCESS":
            continue
        sources = item.get("sources")
        if not isinstance(sources, list) or not sources:
            continue
        known_urls = {
            url.strip()
            for source in sources
            if isinstance(source, dict)
            for url in (source.get("url"), source.get("canonical_url"))
            if _http_url(url)
        }
        source_publishers = {
            url.strip(): str(
                source.get("publisher") or "Unknown"
            ).casefold()
            for source in sources
            if isinstance(source, dict)
            for url in (source.get("url"), source.get("canonical_url"))
            if _http_url(url)
        }
        source_count = len({
            str(source.get("publisher") or "Unknown").casefold()
            for source in sources
            if isinstance(source, dict) and _http_url(source.get("url"))
        })
        source_urls = item.get("source_urls")
        if (
            not known_urls
            or not isinstance(source_urls, list)
            or any(
                not isinstance(url, str) or url not in known_urls
                for url in source_urls
            )
        ):
            continue
        try:
            validated = validate_research_output(
                {
                    "research_brief": item.get("research_brief"),
                    "video_script": item.get("video_script")
                },
                known_urls,
                source_count=source_count,
                source_publishers=source_publishers,
                trusted_topic=item.get("topic"),
                trusted_headline=item.get("headline")
            )
        except ValueError:
            continue

        result = dict(item)
        result.update(validated)
        result["sources"] = [
            source for source in sources
            if isinstance(source, dict) and _http_url(source.get("url"))
        ]
        if not result["sources"]:
            continue
        result["source_count"] = len({
            str(source.get("publisher") or "Unknown").casefold()
            for source in result["sources"]
        })
        result["research_scope"] = (
            "multi_source" if result["source_count"] > 1 else "single_source"
        )
        result["source_urls"] = source_urls
        successful.append(result)
    return successful


def _cite(urls, source_indexes):
    links = []
    seen = set()
    for url in urls:
        if url in seen or url not in source_indexes:
            continue
        seen.add(url)
        index = source_indexes[url]
        links.append(
            f'<link href={quoteattr(url)} color="#246B9E">[{index}]</link>'
        )
    return " ".join(links)


def _source_index(sources):
    indexes = {}
    for index, source in enumerate(sources, start=1):
        for url in (source.get("url"), source.get("canonical_url")):
            if _http_url(url):
                indexes[url.strip()] = index
    return indexes


def _claim_paragraph(claim, source_indexes, styles):
    statement = escape(claim["statement"])
    citations = _cite(claim["source_urls"], source_indexes)
    return (
        f"<b>{statement}</b> {citations}",
        styles["KeyPoint"]
    )


def _source_paragraphs(sources, styles):
    from reportlab.platypus import Paragraph

    story = []
    for index, source in enumerate(sources, start=1):
        publisher = escape(str(source.get("publisher") or "Unknown publisher"))
        title = escape(str(source.get("title") or "Untitled source"))
        published = escape(str(source.get("published") or "Date not provided"))
        url = source.get("url", "")
        link = (
            f'<link href={quoteattr(url)} color="#246B9E">'
            f"{escape(url)}</link>"
            if _http_url(url)
            else "No valid source URL"
        )
        canonical = source.get("canonical_url")
        canonical_text = ""
        if _http_url(canonical) and canonical != url:
            canonical_text = (
                f' &nbsp;|&nbsp; Canonical: <link href={quoteattr(canonical)} '
                f'color="#246B9E">{escape(canonical)}</link>'
            )
        story.append(Paragraph(
            f"<b>[{index}] {publisher}</b> &nbsp;|&nbsp; "
            f"{published}<br/>{title}<br/>{link}{canonical_text}",
            styles["SmallMeta"]
        ))
    return story


def _brief_flowables(item, styles):
    from reportlab.platypus import Paragraph, Spacer

    brief = item["research_brief"]
    script = item["video_script"]
    sources = item["sources"]
    source_indexes = _source_index(sources)
    story = [
        Paragraph(escape(brief["headline"]), styles["ArticleHeadline"]),
        Paragraph(
            f"{escape(item['research_scope'].replace('_', ' ').title())}"
            f" &nbsp;|&nbsp; {item['source_count']} "
            f"{'publisher' if item['source_count'] == 1 else 'publishers'}"
            f" &nbsp;|&nbsp; Confidence: {brief['confidence']:.0%}",
            styles["SmallMeta"]
        )
    ]

    quality = item.get("content_quality", {})
    if isinstance(quality, dict):
        quality_label = escape(
            str(quality.get("label") or "Unreported content quality")
            .replace("_", " ")
            .title()
        )
        story.append(Paragraph(
            f"<b>Content quality:</b> {quality_label} "
            f"({quality.get('sources_with_full_articles', 0)} full article(s); "
            f"{quality.get('sources_using_fallback', 0)} fallback source(s))",
            styles["SmallMeta"]
        ))

    for title, field in _BRIEF_SECTIONS:
        claims = brief[field]
        if not claims:
            continue
        story.append(Paragraph(escape(title), styles["CoverSection"]))
        for claim in claims:
            text, style = _claim_paragraph(claim, source_indexes, styles)
            story.append(Paragraph(text, style, bulletText="\u2022"))

    story.append(Paragraph(
        "AI-generated analysis (interpretation, not a reported fact)",
        styles["CoverSection"]
    ))
    story.append(Paragraph(
        escape(
            brief["ai_generated_analysis"]
            or "No additional analysis was provided."
        ),
        styles["BodyCopy"]
    ))

    comparison = brief["cross_source_comparison"]
    for label, field in (
        ("Agreements across sources", "agreements"),
        ("Differences across sources", "differences")
    ):
        if not comparison[field]:
            continue
        story.append(Paragraph(label, styles["CoverSection"]))
        for claim in comparison[field]:
            text, style = _claim_paragraph(claim, source_indexes, styles)
            story.append(Paragraph(text, style, bulletText="\u2022"))
    if comparison["missing_information"]:
        story.append(Paragraph(
            "Information not covered in supplied content",
            styles["CoverSection"]
        ))
        for item in comparison["missing_information"]:
            citations = _cite(item["source_urls"], source_indexes)
            story.append(Paragraph(
                f"{escape(item['detail'])} {citations}",
                styles["KeyPoint"],
                bulletText="\u2022"
            ))

    cross_check = item.get("cross_check", {})
    if isinstance(cross_check, dict):
        differences = cross_check.get("metadata_differences", [])
        if isinstance(differences, list) and differences:
            story.append(Paragraph(
                "Source metadata differences",
                styles["CoverSection"]
            ))
            for difference in differences:
                if not isinstance(difference, dict):
                    continue
                field = escape(str(difference.get("field", "Metadata")))
                values = difference.get("values", [])
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, dict):
                        continue
                    story.append(Paragraph(
                        f"<b>{field}:</b> "
                        f"{escape(str(value.get('publisher', 'Source')))} — "
                        f"{escape(str(value.get('value', '')))}",
                        styles["KeyPoint"],
                        bulletText="\u2022"
                    ))
        missing_metadata = cross_check.get("missing_metadata", [])
        if isinstance(missing_metadata, list) and missing_metadata:
            story.append(Paragraph(
                "Missing source metadata",
                styles["CoverSection"]
            ))
            for entry in missing_metadata:
                if not isinstance(entry, dict):
                    continue
                publisher = escape(str(entry.get("publisher", "Source")))
                fields = entry.get("missing_fields", [])
                if not isinstance(fields, list):
                    continue
                story.append(Paragraph(
                    f"{publisher}: "
                    f"{escape(', '.join(map(str, fields)) or 'not specified')}",
                    styles["KeyPoint"],
                    bulletText="\u2022"
                ))
        agreement = cross_check.get("claim_agreement")
        if isinstance(agreement, str) and agreement:
            story.append(Paragraph(
                f"<b>Cross-check status:</b> {escape(str(agreement))}",
                styles["SmallMeta"]
            ))

    story.extend([
        Paragraph("Sources", styles["CoverSection"]),
        *_source_paragraphs(sources, styles),
        Spacer(1, 6),
        Paragraph("Video-ready script", styles["SectionHeading"]),
        Paragraph(
            f"<b>Suggested title:</b> "
            f"{escape(script['suggested_title'])}<br/>"
            f"<b>Thumbnail:</b> {escape(script['thumbnail_text'])}<br/>"
            f"<b>Target:</b> {script['target_duration_seconds']} seconds",
            styles["BodyCopy"]
        )
    ])

    for label, field in (
        ("Opening hook", "opening_hook"),
        ("Introduction", "introduction"),
        ("Main explanation", "main_explanation"),
        ("Why viewers should care", "why_viewers_should_care"),
        ("Closing", "closing")
    ):
        story.append(Paragraph(
            f"<b>{label}:</b> {escape(script[field])}",
            styles["BodyCopy"]
        ))

    story.append(Paragraph(
        "Script developments and examples",
        styles["CoverSection"]
    ))
    for field in ("important_developments", "practical_examples"):
        for claim in script[field]:
            text, style = _claim_paragraph(claim, source_indexes, styles)
            story.append(Paragraph(text, style, bulletText="\u2022"))

    story.append(Paragraph("Scene-by-scene narration", styles["CoverSection"]))
    for scene in script["scenes"]:
        scene_sources = _cite(scene["source_urls"], source_indexes)
        story.append(Paragraph(
            f"<b>Scene {scene['scene_number']} "
            f"({scene['duration_seconds']} sec)</b> "
            f"{escape(scene['narration'])} {scene_sources}<br/>"
            f"<b>Visual:</b> {escape(scene['visual_suggestion'])}<br/>"
            f"<b>On-screen text:</b> {escape(scene['on_screen_text'])}",
            styles["BodyCopy"]
        ))
    return story


def build_research_report_story(records, edition, report_date):
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )

    from .styles import NAVY, PALE_BLUE, RULE

    styles = build_styles()
    briefs = successful_research_briefs(records)
    sources = {
        str(source.get("publisher") or "Unknown publisher")
        for item in briefs
        for source in item["sources"]
    }
    story = [
        Spacer(1, 0.8 * inch),
        Paragraph("AI Daily Intelligence", styles["CoverTitle"]),
        Paragraph(
            f"Research briefing &nbsp;|&nbsp; "
            f"{escape(report_date.isoformat())} &nbsp;|&nbsp; "
            f"Edition {edition:02d}",
            styles["CoverSubtitle"]
        ),
        Spacer(1, 0.15 * inch)
    ]
    metrics = Table(
        [
            [
                Paragraph("<b>Research briefs</b>", styles["SmallMeta"]),
                Paragraph("<b>Source publishers</b>", styles["SmallMeta"]),
                Paragraph("<b>Video scripts</b>", styles["SmallMeta"])
            ],
            [str(len(briefs)), str(len(sources)), str(len(briefs))]
        ],
        colWidths=[2.0 * inch] * 3,
        hAlign="LEFT"
    )
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
        Paragraph("Daily AI news overview", styles["SectionHeading"])
    ])
    for item in briefs:
        brief = item["research_brief"]
        first_claims = brief["what_happened"][:2]
        summary = " ".join(claim["statement"] for claim in first_claims)
        sources_index = _source_index(item["sources"])
        citations = " ".join(
            _cite(claim["source_urls"], sources_index)
            for claim in first_claims
        )
        story.append(Paragraph(
            f"<b>{escape(brief['headline'])}</b><br/>"
            f"{escape(summary)} {citations}",
            styles["Highlight"]
        ))

    story.append(PageBreak())
    story.append(Paragraph("Research briefs and video scripts", styles["SectionHeading"]))
    for index, item in enumerate(briefs):
        if index:
            story.append(PageBreak())
        story.extend(_brief_flowables(item, styles))

    return story


def build_daily_programme_story(programme, edition, report_date):
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )

    from ..research.daily_programme import validate_daily_programme_record
    from .styles import NAVY, PALE_BLUE, RULE

    validated = validate_daily_programme_record(programme)
    if validated["programme_date"] != report_date.isoformat():
        raise ValueError(
            "Daily programme date does not match the requested PDF report date."
        )
    styles = build_styles()
    stories = sorted(
        validated["stories"],
        key=lambda item: (
            {"High": 0, "Medium": 1, "Low": 2}[item["importance"]],
            item["topic"].casefold()
        )
    )
    unique_sources = {
        source["url"]
        for item in stories
        for source in item["sources"]
    }
    story = [
        Spacer(1, 0.8 * inch),
        Paragraph("AI Daily Intelligence", styles["CoverTitle"]),
        Paragraph(
            f"Daily AI news programme &nbsp;|&nbsp; "
            f"{escape(report_date.isoformat())} &nbsp;|&nbsp; "
            f"Edition {edition:02d}",
            styles["CoverSubtitle"]
        ),
        Paragraph(
            escape(validated["overall_headline"]),
            styles["SectionHeading"]
        ),
        Paragraph(
            escape(validated["executive_summary"]),
            styles["BodyCopy"]
        )
    ]
    metrics = Table(
        [
            [
                Paragraph("<b>Important stories</b>", styles["SmallMeta"]),
                Paragraph("<b>Original sources</b>", styles["SmallMeta"]),
                Paragraph("<b>Daily programmes</b>", styles["SmallMeta"])
            ],
            [str(len(stories)), str(len(unique_sources)), "2"]
        ],
        colWidths=[2.0 * inch] * 3,
        hAlign="LEFT"
    )
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
        Paragraph("Today's important developments", styles["SectionHeading"])
    ])
    for item in stories:
        publishers = ", ".join(sorted({
            str(source["publisher"]) for source in item["sources"]
        }))
        source_indexes = _source_index(item["sources"])
        story.append(Paragraph(
            f"{escape(item['headline'])}",
            styles["ArticleHeadline"]
        ))
        story.append(Paragraph(
            f"<b>Topic:</b> {escape(item['topic'])} &nbsp;|&nbsp; "
            f"<b>Importance:</b> {item['importance']} &nbsp;|&nbsp; "
            f"<b>Coverage:</b> {item['source_scope'].replace('_', ' ')} "
            f"({escape(publishers)}) &nbsp;|&nbsp; "
            f"<b>Confidence:</b> {item['confidence']:.0%}",
            styles["SmallMeta"]
        ))
        quality = item["content_quality_details"]
        content_sources = ", ".join(quality["source_types"])
        story.append(Paragraph(
            f"<b>Content basis:</b> {escape(content_sources)} "
            f"({quality['full_article_count']} full article(s); "
            f"{quality['fallback_count']} fallback source(s))",
            styles["SmallMeta"]
        ))
        story.append(Paragraph(escape(item["summary"]), styles["BodyCopy"]))
        story.append(Paragraph("Key facts", styles["CoverSection"]))
        for claim in item["key_facts"]:
            text, style = _claim_paragraph(claim, source_indexes, styles)
            story.append(Paragraph(text, style, bulletText="\u2022"))
        story.append(Paragraph(
            f"<b>Why it matters:</b> {escape(item['why_it_matters'])}",
            styles["BodyCopy"]
        ))
        comparison = item["cross_source_comparison"]
        for label, field in (
            ("Agreements", "agreements"),
            ("Contradictions", "contradictions"),
            ("Additional details", "additional_details")
        ):
            if not comparison[field]:
                continue
            story.append(Paragraph(escape(label), styles["CoverSection"]))
            for claim in comparison[field]:
                text, style = _claim_paragraph(claim, source_indexes, styles)
                story.append(Paragraph(text, style, bulletText="\u2022"))
        story.extend([
            Paragraph("Original sources", styles["CoverSection"]),
            *_source_paragraphs(item["sources"], styles)
        ])

    story.append(PageBreak())
    story.append(Paragraph("Combined daily news programmes", styles["SectionHeading"]))
    for key, label in (
        ("one_minute", "1-minute AI News Update"),
        ("three_minute", "3-minute AI News Bulletin")
    ):
        programme = validated["programmes"][key]
        story.extend([
            Paragraph(escape(label), styles["CoverSection"]),
            Paragraph(
                f"<b>{escape(programme['title'])}</b> &nbsp;|&nbsp; "
                f"About {programme['duration_seconds']} seconds &nbsp;|&nbsp; "
                f"{programme['word_count']} words",
                styles["SmallMeta"]
            ),
            Paragraph(escape(programme["narration"]), styles["BodyCopy"]),
            Paragraph(
                "<b>Stories covered:</b> "
                + escape(", ".join(
                    next(
                        item["headline"]
                        for item in stories
                        if item["story_id"] == story_id
                    )
                    for story_id in programme["story_ids"]
                )),
                styles["SmallMeta"]
            ),
            Paragraph("Script source links", styles["SmallMeta"])
        ])
        programme_source_records = [
            source
            for item in stories
            for source in item["sources"]
            if source["url"] in programme["source_urls"]
        ]
        unique_programme_sources = []
        seen_urls = set()
        for source in programme_source_records:
            if source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                unique_programme_sources.append(source)
        story.extend(_source_paragraphs(unique_programme_sources, styles))

    return story
