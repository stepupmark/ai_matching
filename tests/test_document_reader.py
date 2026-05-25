"""Tests for the Document Reader service (PDF, DOCX, TXT extraction)."""

from __future__ import annotations

import pytest

from app.services.document_reader import (
    extract_text,
    _extract_txt,
    _clean_text,
    UnsupportedFormatError,
    DocumentReadError,
)


class TestTxtExtraction:
    def test_extract_utf8(self) -> None:
        text = "Hello, world!\nThis is a resume."
        result = _extract_txt(text.encode("utf-8"))
        assert "Hello" in result
        assert "resume" in result

    def test_extract_empty(self) -> None:
        result = _extract_txt(b"")
        assert result == ""

    def test_extract_unicode(self) -> None:
        text = "Résumé with spécial chàracters"
        result = _extract_txt(text.encode("utf-8"))
        assert "Résumé" in result


class TestCleanText:
    def test_remove_null_bytes(self) -> None:
        assert _clean_text("foo\x00bar") == "foobar"

    def test_normalize_line_endings(self) -> None:
        assert _clean_text("line1\r\nline2\rline3") == "line1\nline2\nline3"

    def test_collapse_blank_lines(self) -> None:
        assert _clean_text("a\n\n\n\nb") == "a\n\nb"


class TestExtractTextDispatch:
    def test_unsupported_format(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            extract_text(b"some data", "file.csv")

    def test_unsupported_extension(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            extract_text(b"some data", "file.xyz")

    def test_txt_via_dispatch(self) -> None:
        text = "Python developer with 5 years experience"
        result = extract_text(text.encode("utf-8"), "resume.txt")
        assert "Python" in result
        assert "5 years" in result


class TestFileSizeLimit:
    def test_oversized_file(self) -> None:
        from app.services.document_reader import MAX_FILE_SIZE
        huge_data = b"x" * (MAX_FILE_SIZE + 1)
        with pytest.raises(DocumentReadError, match="exceeds maximum size"):
            extract_text(huge_data, "file.txt")
