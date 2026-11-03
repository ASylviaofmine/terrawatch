"""Build the Erasmus Mundus CV as a .docx.

    python tools/make_cv_docx.py

Writes docs/CV_Haiyu_Li_Erasmus.docx.

The content lives in CV_ERASMUS.md so it can be reviewed as plain text;
this script only handles layout. Two rules that are not negotiable for a
European academic CV:

- one column, never two
- no photograph

Page budget is two pages, so the type is 10.5pt with tight but readable
spacing, and the margins are 2 cm.
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "CV_Haiyu_Li_Erasmus.docx"

INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
MUTED = RGBColor(0x4A, 0x4A, 0x4A)
LINK = RGBColor(0x0B, 0x53, 0x94)

BODY = "Calibri"
HEAD = "Calibri"


def set_font(run, size=10.5, bold=False, color=INK, italic=False, name=BODY):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = name
    # East-Asian font, so a stray Chinese character renders instead of tofu
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), "微软雅黑")


def para(doc, space_before=0, space_after=2, line=1.0, align=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    return p


def section(doc, title):
    p = para(doc, space_before=8, space_after=3)
    r = p.add_run(title.upper())
    set_font(r, size=11, bold=True, color=ACCENT)
    # a heading must never be the last thing on a page
    p.paragraph_format.keep_with_next = True
    # a thin rule under the heading reads as a section break without a box
    pPr = p._element.get_or_add_pPr()
    pbdr = pPr.makeelement(qn("w:pBdr"), {})
    bottom = pbdr.makeelement(qn("w:bottom"), {})
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "2")
    bottom.set(qn("w:color"), "B8C4D4")
    pbdr.append(bottom)
    pPr.append(pbdr)
    return p


def entry(doc, left, right):
    """A 'Title ......... Date' line with the date right-aligned via a tab."""
    p = para(doc, space_before=3, space_after=1)
    pf = p.paragraph_format
    pf.tab_stops.add_tab_stop(Cm(17.0), WD_ALIGN_PARAGRAPH.RIGHT)
    r = p.add_run(left)
    set_font(r, size=10.5, bold=True)
    r2 = p.add_run("\t" + right)
    set_font(r2, size=10, color=MUTED)
    return p


def sub(doc, text, size=10, italic=False, color=MUTED, space_after=1):
    p = para(doc, space_after=space_after)
    r = p.add_run(text)
    set_font(r, size=size, italic=italic, color=color)
    return p


def bullet(doc, text, size=10, indent=0.45):
    p = para(doc, space_after=1.5, line=1.0)
    pf = p.paragraph_format
    pf.left_indent = Cm(indent + 0.35)
    pf.first_line_indent = Cm(-0.35)
    r = p.add_run("•  " + text)
    set_font(r, size=size)
    return p


def build():
    doc = Document()

    sec = doc.sections[0]
    # A4, not Letter: European applications are read by people who print A4,
    # and a Letter page either gets rescaled (and looks wrong) or fails a
    # format check.
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(1.8)
    sec.bottom_margin = Cm(1.6)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)

    style = doc.styles["Normal"]
    style.font.name = BODY
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    # ---------------- header ----------------
    p = para(doc, space_after=1)
    r = p.add_run("HAIYU LI")
    set_font(r, size=19, bold=True, color=ACCENT)

    p = para(doc, space_after=1)
    r = p.add_run(
        "Jinling Institute of Technology (金陵科技学院) · Nanjing, China"
    )
    set_font(r, size=10, color=MUTED)

    p = para(doc, space_after=1)
    for text, bold in [
        ("B.Eng. Automation Engineering", True),
        ("  (Sep 2022 – Jun 2027, expected)  ·  Student ID 2217011031  ·  ", False),
        ("asylviaofmine@outlook.com", False),
        ("  ·  ", False),
        ("github.com/ASylviaofmine", False),
    ]:
        r = p.add_run(text)
        set_font(r, size=10, bold=bold, color=LINK if "github" in text or "@" in text else INK)

    # ---------------- education ----------------
    section(doc, "Education")
    entry(doc, "Jinling Institute of Technology — B.Eng. Automation Engineering", "Nanjing, China")
    sub(doc, "Sep 2022 – Jun 2027 (expected) · 145 credits completed", color=MUTED, size=9.5)
    bullet(
        doc,
        "Core coursework: Automatic Control Principles, Digital and Analog "
        "Electronics, Microcomputer Principles and Applications, Mathematics Modelling",
    )
    bullet(
        doc,
        "Mathematics Modelling: 96 / 100  ·  National Mathematical Contest in "
        "Modelling, Honorable Mention (team 2516158)",
    )

    # ---------------- projects ----------------
    section(doc, "Selected Projects")
    sub(
        doc,
        "All four are self-contained, independently built and published open-source "
        "with evaluation harnesses and reproducible results.",
        italic=True,
        size=9.5,
        space_after=3,
    )

    entry(doc, "TerraWatch — Satellite Surface Change Detection", "github.com/ASylviaofmine/terrawatch")
    sub(doc, "Python · PyTorch · NumPy · OpenCV  |  Individual research project, Oct 2026", size=9.5)
    bullet(
        doc,
        "Deep-learning system that takes two satellite acquisitions of the same "
        "ground months apart and produces a per-pixel 7-class change map, "
        "distinguishes human-made change (deforestation, urban expansion, "
        "excavation, new road) from natural change (flooding, wildfire scar), and "
        "writes an automated incident report.",
    )
    bullet(
        doc,
        "Designed the procedural remote-sensing data generator with exact "
        "pixel-level ground truth, making every reported metric a measurement "
        "rather than an estimate.",
    )
    bullet(
        doc,
        "Change IoU 0.905, pixel accuracy 99.3%, and 88.1% accuracy at "
        "separating human-made from natural change, measured on 200 scenes "
        "generated from a seed range disjoint from training.",
    )
    bullet(
        doc,
        "Diagnosed three label-level data defects by measurement — excavation, "
        "flooding and urban expansion recovered from IoU 0.03 / 0.00 / 0.07 to "
        "0.57 / 0.64 / 0.61 — showing the failures stemmed from class "
        "separability in the data rather than model capacity.",
    )
    bullet(
        doc,
        "Re-implemented the forward pass in NumPy only for deployment, with a "
        "self-test asserting agreement with the PyTorch model to < 1e-4; shipped "
        "a live web application and a 22-second captioned demo video.",
    )

    entry(doc, "Nemotron-Supervised A* + PID Navigation Agent", "github.com/ASylviaofmine/nemotron-nav-agent")
    sub(doc, "Python · PyTorch  |  Individual project", size=9.5)
    bullet(
        doc,
        "Differential-drive robot that plans with A* on an occupancy grid and "
        "tracks the path with a proportional controller, under the supervision "
        "of an NVIDIA Nemotron model served from Nebius Token Factory — a "
        "language model driving a planning-and-control loop rather than a "
        "scripted demo.",
    )

    entry(doc, "Steel Surface Defect Detection with YOLOv8", "github.com/ASylviaofmine/defect-detection")
    sub(doc, "Python · PyTorch · OpenCV  |  Individual project", size=9.5)
    bullet(
        doc,
        "YOLOv8 detector for steel-surface defects, benchmarked against a "
        "classical-CV baseline that was deliberately implemented first, so the "
        "comparison is defensible rather than assumed.",
    )

    entry(doc, "2D Mobile-Robot Navigation Simulator", "github.com/ASylviaofmine/robot-nav-sim")
    sub(doc, "Python  |  Individual project", size=9.5)
    bullet(
        doc,
        "Grid-based path planning with feedback control, written from scratch "
        "with no robotics framework, used as the substrate for the navigation "
        "work above.",
    )

    # ---------------- competitions ----------------
    section(doc, "Competitions and Awards")
    awards = [
        (
            "ORCA CODERS: ORCATHON 2026 (Dev Harbour, online, 654 participants)",
            "Certificate of Participation",
            "Oct 2026",
        ),
        (
            "Chinese Mathematics Competitions in Higher Education (华数杯)",
            "First Prize (national final)",
            "2025",
        ),
        (
            "Mathematical Contest in Modelling (MCM/ICM, USA)",
            "Honourable Mention — team 2516158",
            "2025",
        ),
        (
            "International English Vocabulary Competition",
            "First Prize",
            "2024",
        ),
        (
            "Debate Competition, School of Intelligent Control",
            "4th place",
            "2024",
        ),
    ]
    for name, result, date in awards:
        p = para(doc, space_after=1)
        pf = p.paragraph_format
        pf.left_indent = Cm(0.35)
        pf.first_line_indent = Cm(-0.35)
        pf.tab_stops.add_tab_stop(Cm(17.0), WD_ALIGN_PARAGRAPH.RIGHT)
        r = p.add_run("•  " + name + " — " + result)
        set_font(r, size=10)
        r2 = p.add_run("\t" + date)
        set_font(r2, size=9.5, color=MUTED)

    sub(
        doc,
        "ORCATHON 2026 project: TerraWatch — source code, live deployment and "
        "demo video publicly available (github.com/ASylviaofmine/terrawatch).",
        size=9.5,
        space_after=2,
    )

    # ---------------- skills ----------------
    section(doc, "Technical Skills")
    p = para(doc, space_after=1)
    # the four lines are one block: splitting them across a page break
    # leaves "Programming" on one page and "Domains" on the next, which
    # reads as a mistake
    p.paragraph_format.keep_together = True
    for i, (label, body) in enumerate([
        ("Programming: ", "Python (NumPy, PyTorch, OpenCV, Matplotlib), C/C++ (basics), SQL, Bash"),
        ("Frameworks & tools: ", "PyTorch, Gradio, Git/GitHub, Linux, ROS (fundamentals), Docker (basics)"),
        ("Domains: ", "Computer vision, remote-sensing image analysis, deep learning for per-pixel semantic segmentation, robot path planning (A*, PID), multi-agent simulation, LLM APIs"),
        ("Data: ", "Synthetic data generation with pixel-exact ground truth, evaluation protocol design (held-out seed ranges; IoU / precision / recall / F1), confusion-matrix analysis, Excel for statistical analysis and reporting"),
    ]):
        # a soft break, not a new paragraph: the four labels stay one block
        # but each starts on its own line
        if i:
            p.runs[-1].add_break()
        r = p.add_run(label)
        set_font(r, size=10, bold=True)
        r2 = p.add_run(body)
        set_font(r2, size=10)

    # ---------------- languages ----------------
    section(doc, "Languages")
    p = para(doc, space_after=1)
    r = p.add_run("Chinese (Mandarin) — native")
    set_font(r, size=10)
    p = para(doc, space_after=1)
    for label, body in [
        ("English — ", "professional working proficiency; technical reading and writing (CET-4)"),
    ]:
        r = p.add_run(label)
        set_font(r, size=10, bold=True)
        r2 = p.add_run(body)
        set_font(r2, size=10)
    sub(
        doc,
        "[ Replace this line with the actual test result once issued, e.g. "
        "\"IELTS 6.5 (L 6.5 / R 7.0 / W 6.0 / S 6.0)\" — the consortiums "
        "require a standard test score and CET-4 is not accepted. ]",
        size=9,
        italic=True,
        color=RGBColor(0xA0, 0x30, 0x30),
    )

    # ---------------- leadership ----------------
    section(doc, "Leadership and Activities")
    entry(doc, "President, University Running Club", "Sep 2023 – Jun 2025")
    bullet(doc, "Organised four campus running events end to end; managed 80+ members, "
                "from recruitment through to day-of execution.")
    entry(doc, "Media Account Lead", "Sep 2023 – present")
    bullet(doc, "Owned content strategy, SEO topic research and audience analysis; grew to "
                "700+ followers with a peak article at 55,000 views and a 14% "
                "follow-conversion rate, tracked and iterated in Excel.")
    entry(doc, "Volunteer — World Youth Forum (Outstanding Volunteer)", "2024 – 2025")
    sub(doc, "VR Industry Expo 2026 · Leiyin Temple, Dunhuang", size=9.5)
    bullet(doc, "Half-month of volunteer service at a Dunhuang temple; recognised as an "
                "Outstanding Volunteer at the World Youth Forum.")

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
