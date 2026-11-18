"""Unit tests for the pure chat-attachment validator (REQ-006..008).

Docker-free contract tests for ``decode_and_validate_attachments``: the
attachment limits come from the Settings-validated env (REQ-092), the branch
coverage mirrors the behavioral validation suite (validation/vision/documents)
so the CRAP gate is verifiable WITHOUT the Docker-backed stack. Every invalid
payload maps to an explicit 422 ``AttachmentValidationError``; every valid
payload decodes strictly with a fresh unique id (EDG-006).
"""

import base64

import pytest

from src.domain.entities.attachment import AttachmentKind, AttachmentLimits
from src.domain.errors.attachment import AttachmentLimitError, AttachmentValidationError
from src.domain.services.chat.attachment_validator import decode_and_validate_attachments

_B64 = base64.b64encode(b"payload-bytes").decode()
_PDF_B64 = base64.b64encode(b"%PDF-1.7 stub document").decode()

_LIMITS = AttachmentLimits(
    max_count=10,
    max_total_bytes=50 * 1024 * 1024,
    max_image_bytes=10 * 1024 * 1024,
    max_document_bytes=50 * 1024 * 1024,
)


def _att(filename: str, content_type: str, content_base64: str = _B64) -> dict:
    return {"filename": filename, "content_type": content_type, "content_base64": content_base64}


@pytest.mark.parametrize(
    "payload",
    [
        [_att("capture.png", "image/png", "not-valid-base64!!!")],  # EC-001
        [_att("capture.png", "image/png", "")],  # empty payload
        [_att("capture.png", "image/png", None)],  # non-str payload
    ],
)
def test_strict_base64_violations_are_rejected(payload):
    with pytest.raises(AttachmentValidationError):
        decode_and_validate_attachments(payload, _LIMITS)


def test_too_many_attachments_are_rejected():
    payload = [_att(f"a{i}.png", "image/png") for i in range(11)]
    with pytest.raises(AttachmentLimitError):
        decode_and_validate_attachments(payload, _LIMITS)


@pytest.mark.parametrize(
    "payload",
    [
        [_att("big.png", "image/png", base64.b64encode(b"\x00" * (10 * 1024 * 1024 + 1)).decode())],
        [_att("big.pdf", "application/pdf", base64.b64encode(b"\x00" * (50 * 1024 * 1024 + 1)).decode())],
        [
            _att("a.pdf", "application/pdf", base64.b64encode(b"\x00" * (30 * 1024 * 1024)).decode()),
            _att("b.pdf", "application/pdf", base64.b64encode(b"\x00" * (25 * 1024 * 1024)).decode()),
        ],  # total > 50 MiB
    ],
)
def test_limit_violations_are_rejected(payload):
    with pytest.raises(AttachmentLimitError):
        decode_and_validate_attachments(payload, _LIMITS)


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("", "image/png"),
        ("   ", "image/png"),
        ("../../etc/passwd.png", "image/png"),
        ("\\windows\\evil.png", "image/png"),
        ("/abs/path.png", "image/png"),
        ("capture.png\nlog-forged", "image/png"),  # control characters
        ("a/.", "image/png"),  # reduces to "." → refused outright
        ("a/..", "image/png"),  # reduces to ".." → refused outright
        (None, "image/png"),  # non-str filename refused before coercion
        ("capture.png", None),
        ("capture.png", ""),
    ],
)
def test_filename_and_mime_violations_are_rejected(filename, content_type):
    with pytest.raises(AttachmentValidationError):
        decode_and_validate_attachments([_att(filename, content_type)], _LIMITS)


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("image.tiff", "image/tiff"),  # EC-002 — non-whitelisted vision MIME
        ("capture.png", "image/gif"),  # incoherent image extension/MIME
        ("report.pdf", "image/png"),  # pdf extension with an image MIME
        ("a.bin", "application/octet-stream"),  # outside the document families
        ("setup.exe", "application/x-msdownload"),  # executable family
        ("data.json", "text/plain"),  # json extension with a text/ MIME — incoherent
        ("notes.txt", "application/json"),  # txt extension with application/ JSON — incoherent
        ("deck.ppt", "application/pdf"),  # office extension with a pdf MIME
        ("memo.doc", "application/msword"),  # legacy .doc pairs with the vnd.ms- family
    ],
)
def test_type_coherence_violations_are_rejected(filename, content_type):
    with pytest.raises(AttachmentValidationError):
        decode_and_validate_attachments([_att(filename, content_type)], _LIMITS)


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("rapport.pdf", "application/pdf"),
        ("notes.txt", "text/plain"),
        ("notes.md", "text/markdown"),
        ("data.csv", "text/csv"),
        ("page.html", "text/html"),
        ("config.json", "application/json"),
        ("report.rtf", "application/rtf"),
        ("recap.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("table.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("deck.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        ("table.xls", "application/vnd.ms-excel"),
        ("deck.ppt", "application/vnd.ms-powerpoint"),
        ("doc.odt", "application/vnd.oasis.opendocument.text"),
        ("sheet.ods", "application/vnd.oasis.opendocument.spreadsheet"),
    ],
)
def test_document_family_is_accepted(filename, content_type):
    attachments = decode_and_validate_attachments([_att(filename, content_type, _PDF_B64)], _LIMITS)
    assert attachments[0].kind == AttachmentKind.DOCUMENT
    assert attachments[0].size_bytes == len(base64.b64decode(_PDF_B64))


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("capture.png", "image/png"),
        ("capture.jpeg", "image/jpeg"),
        ("capture.jpg", "image/jpeg"),
        ("anim.gif", "image/gif"),
        ("photo.webp", "image/webp"),
    ],
)
def test_vision_family_is_accepted(filename, content_type):
    attachments = decode_and_validate_attachments([_att(filename, content_type)], _LIMITS)
    assert attachments[0].kind == AttachmentKind.IMAGE


def test_relative_filenames_are_reduced_to_their_basename():
    attachments = decode_and_validate_attachments([_att("some/dir/photo.png", "image/png")], _LIMITS)
    assert attachments[0].filename == "photo.png"
    assert attachments[0].kind == AttachmentKind.IMAGE


def test_each_attachment_receives_a_unique_id():
    attachments = decode_and_validate_attachments(
        [_att("same.png", "image/png"), _att("same.png", "image/png")], _LIMITS
    )
    assert len({a.id for a in attachments}) == 2


def test_content_is_never_exposed_by_the_model():
    attachments = decode_and_validate_attachments([_att("capture.png", "image/png")], _LIMITS)
    validated = attachments[0]
    assert "content_base64" not in repr(validated)
    assert "content_base64" not in validated.model_dump()
    assert validated.content_base64 == _B64  # original payload preserved untampered


def test_mixed_turn_keeps_declared_order():
    attachments = decode_and_validate_attachments(
        [_att("capture.png", "image/png"), _att("rapport.pdf", "application/pdf", _PDF_B64)], _LIMITS
    )
    assert [a.kind for a in attachments] == [AttachmentKind.IMAGE, AttachmentKind.DOCUMENT]


def test_empty_payload_and_none_are_no_ops():
    assert decode_and_validate_attachments(None, _LIMITS) == []
    assert decode_and_validate_attachments([], _LIMITS) == []
