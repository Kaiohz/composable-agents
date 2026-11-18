"""Pure renderer: mistletoe AST → fpdf2-supported HTML subset.

The builtin ``create_pdf`` tool parses Markdown with mistletoe (REQ-053) and
lays it out with fpdf2's ``write_html``. This module is the bridge: a custom
mistletoe ``BaseRenderer`` emitting ONLY the HTML tags fpdf2 supports:

* headings H1–H6, paragraphs, bold/italic/strikethrough (``<del>``), inline
  code, unordered + ordered lists, blockquotes, code blocks (``<pre>``),
  tables, links, horizontal rules.

``render_image`` is deliberately a no-op (REQ-055): NO remote image is ever
downloaded — the alternative text stands in instead.

fpdf2 core fonts encode Latin-1 only: every text node is coerced to a
Latin-1-safe string (unencodable characters are replaced) so accented names
(EDG-010) keep working and exotic symbols degrade gracefully instead of
crashing the PDF generation.
"""

from html import escape as html_escape
from typing import Any

from mistletoe.base_renderer import BaseRenderer


def latin1_safe(text: str) -> str:
    """Coerce ``text`` to a Latin-1-safe string (replacement on overflow)."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _esc(text: Any) -> str:
    """HTML-escape a text node before handing it to ``write_html``."""
    return html_escape(latin1_safe(str(text)))


def _esc_attr(value: Any) -> str:
    """Escape an HTML attribute value (href/target)."""
    return html_escape(latin1_safe(str(value)), quote=True)


class FpdfHtmlRenderer(BaseRenderer):
    """Render a mistletoe Document into the fpdf2-supported HTML subset."""

    def render_document(self, token: Any) -> str:
        """Render the document body (no html/head/body wrapper for fpdf2)."""
        return "".join(self.render(child) for child in token.children)

    # --- Block-level ---------------------------------------------------------
    def render_paragraph(self, token: Any) -> str:
        return f"<p>{self.render_inner(token)}</p>\n"

    def render_heading(self, token: Any) -> str:
        level = min(max(int(getattr(token, "level", 1)), 1), 6)
        return f"<h{level}>{self.render_inner(token)}</h{level}>\n"

    def render_list(self, token: Any) -> str:
        tag = "ol" if getattr(token, "start", None) is not None else "ul"
        return f"<{tag}>{self.render_inner(token)}</{tag}>\n"

    def render_list_item(self, token: Any) -> str:
        return f"<li>{self.render_inner(token)}</li>\n"

    def render_quote(self, token: Any) -> str:
        return f"<blockquote>{self.render_inner(token)}</blockquote>\n"

    def render_block_code(self, token: Any) -> str:
        text = self._code_text(token)
        return f"<pre>{_esc(text)}</pre>\n"

    def render_table(self, token: Any) -> str:
        # mistletoe splits the table into ``header`` (a TableRow) and ``children``
        # (body rows) — render them directly so header cells become ``th``.
        header = getattr(token, "header", None)
        thead = ""
        if header is not None:
            cells = "".join(f"<th>{self.render_inner(cell)}</th>" for cell in header.children)
            thead = f"<thead><tr>{cells}</tr></thead>\n"
        body = "".join(
            f"<tr>{''.join(f'<td>{self.render_inner(cell)}</td>' for cell in row.children)}</tr>\n"
            for row in getattr(token, "children", []) or []
        )
        return f"<table>{thead}<tbody>{body}</tbody></table>\n"

    def render_thematic_break(self, token: Any) -> str:  # noqa: ARG002 (token unused: fixed marker)
        return "<hr/>\n"

    # --- Inline-level ----------------------------------------------------------
    def render_strong(self, token: Any) -> str:
        return f"<b>{self.render_inner(token)}</b>"

    def render_emphasis(self, token: Any) -> str:
        return f"<i>{self.render_inner(token)}</i>"

    def render_strikethrough(self, token: Any) -> str:
        return f"<del>{self.render_inner(token)}</del>"

    def render_inline_code(self, token: Any) -> str:
        return f"<code>{_esc(self._code_text(token))}</code>"

    def render_line_break(self, token: Any) -> str:
        soft = getattr(token, "soft", False)
        return "\n" if soft else "<br/>\n"

    def render_raw_text(self, token: Any) -> str:
        return _esc(getattr(token, "content", ""))

    def render_escape_sequence(self, token: Any) -> str:
        return _esc(self._code_text(token))

    def render_link(self, token: Any) -> str:
        target = getattr(token, "target", "") or ""
        return f'<a href="{_esc_attr(target)}">{self.render_inner(token)}</a>'

    def render_auto_link(self, token: Any) -> str:
        target = getattr(token, "target", "") or ""
        return f'<a href="{_esc_attr(target)}">{_esc(target)}</a>'

    def render_image(self, token: Any) -> str:
        # REQ-055: NO remote image ever downloaded — alt text stands in.
        return f"[image: {self.render_inner(token)}]"

    # --- helpers -------------------------------------------------------------------
    @staticmethod
    def _code_text(token: Any) -> str:
        """Extract the raw code text of an InlineCode/EscapeSequence/BlockCode.

        mistletoe wraps code payloads in a RawText child; BlockCode carries a
        tuple with a trailing newline. Only ``str`` values are accepted, other
        shapes degrade safely (no crash, no silent markup injection).

        Args:
            token: Any code-bearing mistletoe token.

        Returns:
            The raw code string.
        """
        children = getattr(token, "children", None)
        if isinstance(children, str):
            return children
        if isinstance(children, (tuple, list)):
            parts: list[str] = []
            for child in children:
                content = getattr(child, "content", None)
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(child, str):
                    parts.append(child)
            return "".join(parts).rstrip("\n")
        return ""
