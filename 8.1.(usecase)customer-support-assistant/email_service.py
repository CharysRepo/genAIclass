import smtplib
from models import EmailDetails
from email.mime.text import MIMEText

class EmailService:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = "info@algorithmicaonline.com"
        self.sender_password = ""

    def send_email(self, email_details: EmailDetails):
        msg = MIMEText(email_details.body)
        msg['Subject'] = email_details.subject
        msg['To'] = email_details.to_email

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            print(f"📧 An email sent to {email_details.to_email}")
        except Exception as e:
            print(f"❌ Email error: {e}")
            raise e

if __name__ == "__main__":
    service = EmailService()

    test_data = EmailDetails(
        to_email="algorithmica.desktop@gmail.com",
        subject="ACT Fibernet: Ticket Update",
        body="Hello, Ticket ACT-12345 has been raised."
    )

    print("🚀 Starting email test...")
    service.send_email(test_data)