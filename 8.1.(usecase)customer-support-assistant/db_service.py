from pathlib import Path
import sqlite3
from models import Ticket

class DBService:
    def __init__(self):
        self.db_path  = Path(__file__).parent / "isp_tickets.db"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT, 
                    customer_email TEXT, 
                    issue_type TEXT,
                    priority TEXT, 
                    description TEXT
                )
            ''')
    
    def save_ticket(self, ticket: Ticket):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tickets (customer_id, customer_email, issue_type, priority, description) VALUES (?, ?, ?, ?, ?)",
                (ticket.customer_id, ticket.customer_email, ticket.ticket_type, ticket.priority, ticket.description)
            )

        print(f"✅ Ticket saved to database for Customer: {ticket.customer_id}")

if __name__ == "__main__":
    db = DBService()

    test_ticket = Ticket(
        customer_id="ACT123456789",
        customer_email="testuser@example.com",
        description="Internet is down since morning. LOS light is red.",
        ticket_type="connection",
        priority="high"
    )

    print("🚀 Testing DB Save...")
    db.save_ticket(test_ticket)

    print("\n🔍 Verifying data in 'isp_tickets.db'...")
    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                print(f"Retrieved Row: ID={row[0]}, CustID={row[1]}, Email={row[2]}, Issue={row[3]}, Priority={row[4]}")
            else:
                print("❌ No data found in table.")
    except Exception as e:
        print(f"❌ Verification error: {e}") 
        raise e