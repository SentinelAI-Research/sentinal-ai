from datetime import datetime, timezone


def send_message(
    recipient: str,
    message: str,
) -> dict:
    """
    Simulate an outbound message while recording the
    actual attempted communication.

    No real external message is sent.
    """

    if not recipient.strip():
        raise ValueError("Recipient cannot be empty.")

    if not message.strip():
        raise ValueError("Message cannot be empty.")

    return {
        "status": "sent",
        "recipient": recipient,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }