from __future__ import annotations

import email
import re
from email import policy
from email.message import Message
from pathlib import PurePosixPath

from pydantic import BaseModel, Field


class NormalizedEmailRecord(BaseModel):
    """
    Canonical representation of one Enron email.

    The raw Enron corpus contains extensionless mail files
    organized under maildir/<user>/<folder>/<message>.
    """

    record_id: str
    source_dataset: str = "enron"

    message_id: str | None = None
    date: str | None = None

    sender: str | None = None
    recipients: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    bcc: list[str] = Field(default_factory=list)

    subject: str = ""

    source_path: str
    user: str
    folder: str

    source_text: str

    metadata: dict[str, str] = Field(default_factory=dict)


class EnronNormalizer:
    """
    Normalizes raw Enron email messages into a stable
    dataset-independent representation.
    """

    @staticmethod
    def _split_addresses(value: str | None) -> list[str]:
        if not value:
            return []

        parts = []

        for item in value.split(","):
            cleaned = item.strip()

            if cleaned:
                parts.append(cleaned)

        return parts

    @staticmethod
    def _extract_plain_text(message: Message) -> str:
        """
        Extract readable text from a MIME message.

        Plain-text parts are preferred. If no plain-text part
        exists, text/html is converted to a conservative
        text representation.
        """

        if message.is_multipart():
            plain_parts: list[str] = []
            html_parts: list[str] = []

            for part in message.walk():
                if part.get_content_maintype() == "multipart":
                    continue

                content_type = part.get_content_type()

                if content_type == "text/plain":
                    try:
                        content = part.get_content()
                    except (LookupError, UnicodeDecodeError):
                        payload = part.get_payload(
                            decode=True
                        )

                        if payload is None:
                            continue

                        content = payload.decode(
                            "utf-8",
                            errors="replace",
                        )

                    if isinstance(content, str):
                        plain_parts.append(content)

                elif content_type == "text/html":
                    try:
                        content = part.get_content()
                    except (LookupError, UnicodeDecodeError):
                        payload = part.get_payload(
                            decode=True
                        )

                        if payload is None:
                            continue

                        content = payload.decode(
                            "utf-8",
                            errors="replace",
                        )

                    if isinstance(content, str):
                        html_parts.append(content)

            if plain_parts:
                return "\n\n".join(plain_parts).strip()

            if html_parts:
                return EnronNormalizer._strip_html(
                    "\n\n".join(html_parts)
                )

            return ""

        content_type = message.get_content_type()

        try:
            content = message.get_content()
        except (LookupError, UnicodeDecodeError):
            payload = message.get_payload(
                decode=True
            )

            if payload is None:
                return ""

            content = payload.decode(
                "utf-8",
                errors="replace",
            )

        if not isinstance(content, str):
            return ""

        if content_type == "text/html":
            return EnronNormalizer._strip_html(content)

        return content.strip()

    @staticmethod
    def _strip_html(text: str) -> str:
        text = re.sub(
            r"<script\b[^>]*>.*?</script>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        text = re.sub(
            r"<style\b[^>]*>.*?</style>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _parse_path(source_path: str) -> tuple[str, str]:
        """
        Extract user and folder information from:

        maildir/<user>/<folder>/<message>
        """

        path = PurePosixPath(source_path)
        parts = path.parts

        if len(parts) < 4 or parts[0] != "maildir":
            raise ValueError(
                "Unexpected Enron source path: "
                f"{source_path}"
            )

        user = parts[1]

        folder = "/".join(parts[2:-1])

        if not folder:
            raise ValueError(
                "Enron email path does not contain a folder: "
                f"{source_path}"
            )

        return user, folder

    @classmethod
    def normalize(
        cls,
        raw_bytes: bytes,
        source_path: str,
    ) -> NormalizedEmailRecord:

        if not raw_bytes:
            raise ValueError(
                f"Email '{source_path}' is empty."
            )

        user, folder = cls._parse_path(source_path)

        message = email.message_from_bytes(
            raw_bytes,
            policy=policy.default,
        )

        message_id = message.get("Message-ID")
        date = message.get("Date")
        sender = message.get("From")
        subject = message.get("Subject") or ""

        recipients = cls._split_addresses(
            message.get("To")
        )

        cc = cls._split_addresses(
            message.get("Cc")
        )

        bcc = cls._split_addresses(
            message.get("Bcc")
        )

        source_text = cls._extract_plain_text(
            message
        )

        if not source_text:
            raise ValueError(
                f"Email '{source_path}' contains no readable text body."
            )

        metadata = {
            "x_from": message.get("X-From") or "",
            "x_to": message.get("X-To") or "",
            "x_cc": message.get("X-cc") or "",
            "x_bcc": message.get("X-bcc") or "",
            "x_folder": message.get("X-Folder") or "",
            "x_origin": message.get("X-Origin") or "",
            "x_filename": message.get("X-FileName") or "",
        }

        path = PurePosixPath(source_path)

        record_id = str(path)

        return NormalizedEmailRecord(
            record_id=record_id,
            message_id=message_id,
            date=date,
            sender=sender,
            recipients=recipients,
            cc=cc,
            bcc=bcc,
            subject=subject,
            source_path=source_path,
            user=user,
            folder=folder,
            source_text=source_text,
            metadata=metadata,
        )