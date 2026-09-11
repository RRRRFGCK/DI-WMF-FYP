"""Typeset the audit Markdown without altering thesis or experimental records."""
import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / "output" / "pdf" / "FYP6_DEEP_AUDIT_20260908.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("YaHei", "C:/Windows/Fonts/msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("YaHeiBold", "C:/Windows/Fonts/msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold")

styles = {
    "body": ParagraphStyle("body", fontName="YaHei", fontSize=10.2, leading=16.1,
                           spaceAfter=9, textColor=colors.HexColor("#212121"),
                           wordWrap="CJK", splitLongWords=True),
    "title": ParagraphStyle("title", fontName="YaHeiBold", fontSize=21, leading=28,
                            spaceAfter=19, wordWrap="CJK"),
    "h2": ParagraphStyle("h2", fontName="YaHeiBold", fontSize=15, leading=22,
                         spaceAfter=14, wordWrap="CJK", keepWithNext=True),
    "h3": ParagraphStyle("h3", fontName="YaHeiBold", fontSize=11.5, leading=18,
                         spaceBefore=4, spaceAfter=6, wordWrap="CJK", keepWithNext=True),
    "table": ParagraphStyle("table", fontName="YaHei", fontSize=9.3, leading=14.4,
                            wordWrap="CJK"),
    "note": ParagraphStyle("note", fontName="YaHei", fontSize=8.1, leading=11.6,
                           spaceAfter=3, textColor=colors.HexColor("#555555"),
                           wordWrap="CJK", splitLongWords=True),
    "source": ParagraphStyle("source", fontName="YaHei", fontSize=9.0, leading=13.8,
                             spaceAfter=8, wordWrap="CJK", splitLongWords=True),
}

def markup(text):
    text = html.escape(text)
    text = re.sub(r"\[\^(\d+)\]", r"<super>\1</super>", text)
    text = re.sub(r"(https?://[^\s]+)", r'<link href="\1" color="#303030">\1</link>', text)
    return text

story = []
page_sections = []
raw = (HERE / "FYP6_DEEP_AUDIT.md").read_text(encoding="utf-8")
sections = raw.split("<!-- PAGEBREAK -->")
for si, section in enumerate(sections):
    if si:
        story.append(PageBreak())
    lines = section.strip().splitlines()
    i = 0
    page_sections.append({"section": si+1, "characters": len(section)})
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                values = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", v) for v in values):
                    rows.append([Paragraph(markup(v), styles["table"]) for v in values])
                i += 1
            width = A4[0]-88
            columns = len(rows[0])
            ratios = [0.13, 0.48, 0.39] if si == 0 else (
                [0.19, 0.27, 0.54] if si == 1 else (
                [0.17, 0.36, 0.47] if si == 8 else [0.48, 0.22, 0.30]))
            if columns != 3:
                ratios = [1/columns]*columns
            table = Table(rows, colWidths=[width*r for r in ratios], repeatRows=1,
                          hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAEAEA")),
                ("LINEBELOW", (0,0), (-1,0), 0.6, colors.HexColor("#808080")),
                ("LINEBELOW", (0,1), (-1,-1), 0.3, colors.HexColor("#DDDDDD")),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 7),
                ("RIGHTPADDING", (0,0), (-1,-1), 7),
                ("TOPPADDING", (0,0), (-1,-1), 7),
                ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ]))
            story.extend([table, Spacer(1, 11)])
            continue
        if line.startswith("[^" ):
            notes = []
            while i < len(lines) and lines[i].strip().startswith("[^"):
                note = re.sub(r"^\[\^(\d+)\]:", r"\1.", lines[i].strip())
                notes.append(Paragraph(markup(note), styles["note"]))
                i += 1
            story.append(KeepTogether([Spacer(1,5), HRFlowable(width="100%", thickness=.35,
                                                               color=colors.HexColor("#BBBBBB")),
                                       Spacer(1,5), *notes]))
            continue
        if line.startswith("### "):
            story.append(Paragraph(markup(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            story.append(Paragraph(markup(line[3:]), styles["h2"]))
        elif line.startswith("# "):
            story.append(Paragraph(markup(line[2:]), styles["title"]))
        else:
            parts = [line]
            while i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith(("#", "|", "[^")):
                i += 1
                parts.append(lines[i].strip())
            story.append(Paragraph(markup(" ".join(parts)), styles["source" if si == 9 else "body"]))
        i += 1

def page_number(canvas, doc):
    canvas.setFont("YaHei", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(A4[0]-44, 22, str(doc.page))

doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=44, leftMargin=44,
                        topMargin=39, bottomMargin=36,
                        title="FYP6 论文与实验代码深度审计", author="")
doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
print(json.dumps({"pdf": str(OUTPUT), "sections": page_sections}, ensure_ascii=False))
