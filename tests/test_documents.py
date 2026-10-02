import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from teardown.documents import read_proposal


class DocumentTests(unittest.TestCase):
    def test_word_paragraphs_tables_and_tracked_changes(self):
        xml = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body><w:p><w:r><w:t>Research </w:t></w:r><w:r><w:t>proposal</w:t></w:r></w:p>
        <w:p><w:del><w:r><w:t>Deleted text</w:t></w:r></w:del>
        <w:ins><w:r><w:t>Current hypothesis</w:t></w:r></w:ins></w:p>
        <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Budget</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Six months</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
        </w:body></w:document>'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Proposal.docx"
            with ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", xml)
            self.assertEqual(read_proposal(path), "Research proposal\n\nCurrent hypothesis\n\nBudget | Six months")

    def test_corrupt_word_file_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.docx"
            path.write_text("not a zip")
            with self.assertRaisesRegex(ValueError, "Re-save"):
                read_proposal(path)

    def test_old_word_format_has_conversion_guidance(self):
        with self.assertRaisesRegex(ValueError, "Save as .docx"):
            read_proposal(Path("proposal.doc"))

    def test_text_input_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proposal.md"
            path.write_text("# Hypothesis\n\nA → B", encoding="utf-8")
            self.assertEqual(read_proposal(path), "# Hypothesis\n\nA → B")
