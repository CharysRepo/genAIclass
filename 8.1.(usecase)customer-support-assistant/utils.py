from models import Ticket, EmailDetails

def get_custom_instructions(ticket: Ticket) -> str:
    """Returns specific guidance based on the issue type."""
    if ticket.ticket_type == "connection":
        return (
            "🛠️ **Immediate Troubleshooting Steps:**\n"
            "- Unplug your router for 30 seconds and plug it back in.\n"
            "- Check if all cables are securely connected.\n"
            "- If the lights are red, a technician may be required."
        )
    elif ticket.ticket_type == "billing":
        return (
            "💰 **Billing Information:**\n"
            "- View your latest invoice details at: https://isp.com\n"
            "- Our billing team reviews disputes within 2-3 business days."
        )
    return "ℹ️ Our team will review your inquiry and get back to you shortly."

def get_email_content(ticket: Ticket) -> EmailDetails:
    """Generates email content based on the ticket."""
    instructions = get_custom_instructions(ticket)
    subject = f"[{ticket.priority.upper()}] ActFiber Support - {ticket.customer_id}"
    body = (
        f"Hello {ticket.customer_id},\n\n"
        f"We've received your {ticket.ticket_type} request regarding: \"{ticket.description}\".\n\n"
        f"{instructions}\n\n"
        f"Ticket Reference: {ticket.customer_id}-{ticket.priority}"
    )
    return EmailDetails(
        to_email=ticket.customer_email,
        subject=subject,
        body=body
    )