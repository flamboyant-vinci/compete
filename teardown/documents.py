"""Read proposal text, including the main body of Word .docx files."""

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _text(element: ET.Element) -> str:
    if element.tag in (W + "del", W + "moveFrom"):
        return ""
    if element.tag == W + "t":
        return element.text or ""
    if element.tag == W + "tab":
        return "\t"
    if element.tag in (W + "br", W + "cr"):
        return "\n"
    return "".join(_text(child) for child in element)


def _blocks(element: ET.Element) -> list[str]:
    blocks = []
    for child in element:
        if child.tag == W + "p":
            blocks.append(_text(child))
        elif child.tag == W + "tbl":
            for row in child.findall(W + "tr"):
                cells = [" / ".join(_blocks(cell)) for cell in row.findall(W + "tc")]
                blocks.append(" | ".join(cells))
        elif child.tag not in (W + "del", W + "moveFrom"):
            blocks.extend(_blocks(child))
    return blocks


def read_proposal(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".doc":
        raise ValueError("Older .doc files are not supported. Save as .docx or plain text in Word/LibreOffice first.")
    if suffix != ".docx":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError("Input must be a Word .docx file or UTF-8 text/Markdown") from None
    try:
        with ZipFile(path) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > 10_000_000:
                raise ValueError("Word document body is too large; supply a focused summary")
            root = ET.fromstring(archive.read(info))
        body = root.find(W + "body")
        if body is None:
            raise ValueError("Word document has no readable main body")
        return "\n\n".join(_blocks(body)).strip()
    except (BadZipFile, KeyError, ET.ParseError, RuntimeError, NotImplementedError):
        raise ValueError("Cannot read this .docx file. Re-save it as an unencrypted .docx or export plain text.") from None
