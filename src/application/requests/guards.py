"""Reusable request-boundary validation guards (QA PASS 3, BUG-006…BUG-010).

The PATCH DTOs of the feature registries follow *field-presence* semantics: an
absent field is untouched, and for the fields the spec declares **nullable**
(exactly ``description`` on prompts — Models no longer carry a clearable field,
their editable fields are required on every write) an explicit ``null`` clears
the stored value. An explicit ``null`` for every other field is a malformed body
and must be rejected at the request boundary with a clean 422 — never allowed to
reach the database (where a NOT NULL violation would surface as a 5xx with
internal SQL in the message).

``NonNullableFieldsGuard`` implements that boundary once, declaratively: each
request model declares the subset of its editable fields the spec marks
non-nullable and inherits the guard.
"""

from typing import ClassVar, Self

from pydantic import BaseModel, model_validator


class NonNullableFieldsGuard(BaseModel):
    """Reject an explicit ``null`` for fields the spec declares non-nullable.

    Field absence stays valid (PATCH semantics: absent = untouched). A present
    field bound to ``None`` is accepted ONLY for the fields the request model
    excludes from :attr:`NON_NULLABLE_FIELDS` — i.e. the spec-declared
    clearable fields.

    Usage::

        class UpdateXRequest(NonNullableFieldsGuard):
            NON_NULLABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"name"})
            nullable_field: str | None = None
    """

    NON_NULLABLE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="after")
    def _reject_explicit_null_for_non_nullable(self) -> Self:
        """Raise a clean validation error pointing at the offending field."""
        for field_name in type(self).NON_NULLABLE_FIELDS:
            if field_name in self.model_fields_set and getattr(self, field_name) is None:
                raise ValueError(
                    f"'{field_name}' must not be null: it is a required, "
                    "non-nullable field and only the spec-declared nullable fields "
                    "may be cleared with an explicit null"
                )
        return self
