"""
WISDOM Lab - AI 리포트 → Word(.docx) 변환 모듈
마크다운 텍스트를 python-docx로 변환해 bytes를 반환합니다.
"""

import io
import re
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def _set_paragraph_border_bottom(paragraph, color="2E75B6", size=12):
    """단락 하단 테두리 설정 (제목 구분선)"""
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    pBdr.append(bottom)
    pPr.append(pBdr)


def _apply_styles(doc: Document):
    """문서 기본 스타일 설정"""
    style = doc.styles["Normal"]
    font = style.font
    font.name = "맑은 고딕"
    font.size = Pt(10.5)

    # 한글 폰트 설정
    rPr = style.element.get_or_add_rPr()
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    rPr.insert(0, rFonts)


def _add_heading(doc: Document, text: str, level: int):
    """레벨별 제목 추가"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    run = p.add_run(text)
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
        _set_paragraph_border_bottom(p, "2E75B6", 12)
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
    elif level == 3:
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(2)

    # 한글 폰트 적용
    rPr = run._r.get_or_add_rPr()
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    rFonts.set(qn("w:ascii"), "맑은 고딕")
    rPr.insert(0, rFonts)

    return p


def _add_body_paragraph(doc: Document, text: str):
    """본문 단락 추가 (볼드 인라인 처리 포함)"""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0)

    # **bold** 패턴 파싱
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2])
            run.bold = True
        else:
            run = p.add_run(part)

        run.font.size = Pt(10.5)
        rPr = run._r.get_or_add_rPr()
        rFonts = OxmlElement("w:rFonts")
        rFonts.set(qn("w:eastAsia"), "맑은 고딕")
        rFonts.set(qn("w:ascii"), "맑은 고딕")
        rPr.insert(0, rFonts)

    return p


def _add_bullet(doc: Document, text: str, level: int = 0):
    """불릿 항목 추가"""
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.3 + level * 0.25)
    p.paragraph_format.space_after = Pt(2)

    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2])
            run.bold = True
        else:
            run = p.add_run(part)

        run.font.size = Pt(10.5)
        rPr = run._r.get_or_add_rPr()
        rFonts = OxmlElement("w:rFonts")
        rFonts.set(qn("w:eastAsia"), "맑은 고딕")
        rFonts.set(qn("w:ascii"), "맑은 고딕")
        rPr.insert(0, rFonts)

    return p


def markdown_to_docx_bytes(
    md_text: str,
    title: str = "WISDOM Lab 분석 리포트",
    subject: str = "",
    doc_type_label: str = "연구보고서",
) -> bytes:
    """
    마크다운 텍스트를 Word 문서 bytes로 변환합니다.

    Parameters
    ----------
    md_text : str   AI가 생성한 마크다운 리포트 텍스트
    title   : str   문서 제목
    subject : str   연구 주제
    doc_type_label : str  문서 유형 레이블

    Returns
    -------
    bytes  Word(.docx) 파일 바이트
    """
    doc = Document()

    # 페이지 여백 설정 (A4)
    section = doc.sections[0]
    section.page_width  = int(11906)   # A4 너비 (DXA)
    section.page_height = int(16838)   # A4 높이 (DXA)
    section.left_margin   = Inches(1.0)
    section.right_margin  = Inches(1.0)
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)

    _apply_styles(doc)

    # ── 표지 영역 ───────────────────────────────────────────────
    cover_p = doc.add_paragraph()
    cover_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cover_p.paragraph_format.space_before = Pt(36)
    cover_p.paragraph_format.space_after = Pt(6)

    title_run = cover_p.add_run(title)
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    rPr = title_run._r.get_or_add_rPr()
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    rFonts.set(qn("w:ascii"), "맑은 고딕")
    rPr.insert(0, rFonts)

    if subject:
        sub_p = doc.add_paragraph()
        sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub_p.paragraph_format.space_after = Pt(4)
        sub_run = sub_p.add_run(f"주제: {subject}")
        sub_run.font.size = Pt(12)
        sub_run.font.color.rgb = RGBColor(0x44, 0x72, 0xC4)

    type_p = doc.add_paragraph()
    type_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    type_p.paragraph_format.space_after = Pt(24)
    type_run = type_p.add_run(f"[{doc_type_label}]")
    type_run.font.size = Pt(11)
    type_run.font.italic = True
    type_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    doc.add_paragraph()  # 여백

    # ── 본문 파싱 ───────────────────────────────────────────────
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # H1 (#)
        if line.startswith("# ") and not line.startswith("## "):
            _add_heading(doc, line[2:].strip(), 1)

        # H2 (##)
        elif line.startswith("## ") and not line.startswith("### "):
            _add_heading(doc, line[3:].strip(), 2)

        # H3 (###)
        elif line.startswith("### "):
            _add_heading(doc, line[4:].strip(), 3)

        # 불릿 (- 또는 * 또는 숫자.)
        elif re.match(r"^[-*]\s+", line):
            _add_bullet(doc, re.sub(r"^[-*]\s+", "", line).strip(), 0)

        elif re.match(r"^\d+\.\s+", line):
            _add_bullet(doc, re.sub(r"^\d+\.\s+", "", line).strip(), 0)

        # 들여쓰기 불릿 (  - 또는  *)
        elif re.match(r"^\s{2,}[-*]\s+", line):
            _add_bullet(doc, re.sub(r"^\s+[-*]\s+", "", line).strip(), 1)

        # 구분선 (---)
        elif re.match(r"^[-_*]{3,}$", line.strip()):
            p = doc.add_paragraph()
            pPr = p._p.get_or_add_pPr()
            pBdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:space"), "1")
            bottom.set(qn("w:color"), "CCCCCC")
            pBdr.append(bottom)
            pPr.append(pBdr)

        # 빈 줄
        elif line.strip() == "":
            pass  # 빈 줄은 건너뜀

        # 일반 본문
        else:
            _add_body_paragraph(doc, line.strip())

        i += 1

    # ── 푸터: WISDOM Lab ────────────────────────────────────────
    from docx.oxml import OxmlElement as oxEl
    footer = section.footer
    footer_p = footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer_p.add_run("WISDOM Lab · Generated by AI")
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)

    # bytes 반환
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()
