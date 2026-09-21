import datetime
import calendar
from pathlib import Path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from models import DeleteInvite, MeetingInvite, RecurringMeetingInvite, RetrievalByTitleInvite, UpdateInvite, RetrievalByDateInvite

class CalenderService:
    def __init__(self):
        script_location = Path(__file__).parent
        cred_file_path = script_location / 'cred.json'
        SCOPES = ['https://www.googleapis.com/auth/calendar']
        flow = InstalledAppFlow.from_client_secrets_file(cred_file_path, SCOPES)
        creds = flow.run_local_server(port=0)
        self.service = build('calendar', 'v3', credentials=creds)

    def create_google_invite(self, invite: MeetingInvite) -> str:
        try:
            end_time = invite.start_time + datetime.timedelta(minutes=invite.duration_minutes)
            start_str = invite.start_time.strftime('%Y-%m-%dT%H:%M:%S')
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')

            event_body = {
                'summary': invite.title,
                'start': {'dateTime': start_str, 'timeZone': invite.timezone},
                'end': {'dateTime': end_str, 'timeZone': invite.timezone},
                'attendees': [{'email': email} for email in invite.attendees],
            }

            if isinstance(invite, RecurringMeetingInvite):
                recurrence_rule = str(invite.recurrence)
                if not recurrence_rule.startswith("RRULE:"):
                    recurrence_rule = f"RRULE:{recurrence_rule}"
                event_body['recurrence'] = [recurrence_rule]
            print(event_body)

            event = self.service.events().insert(
                calendarId='primary', body=event_body, sendUpdates='none'
            ).execute()
            return f"New invite sent for '{invite.title}'! Link: {event.get('htmlLink')}"
        except Exception as e:
            print(f"Failed to create invite: {str(e)}")
            raise e
        

    def update_google_invite(self, update: UpdateInvite) -> str:
        try:
            event = self.service.events().get(
                calendarId='primary', 
                eventId=update.event_id
            ).execute()
            
            current_attendees = event.get('attendees', [])
            
            if update.action == "add":
                existing_emails = {a.get('email') for a in current_attendees}
                new_list = current_attendees + [
                    {'email': email} for email in update.attendees 
                    if email not in existing_emails
                ]
                msg = f"Added {len(new_list) - len(current_attendees)} attendee(s)."

            elif update.action == "remove":
                new_list = [
                    a for a in current_attendees 
                    if a.get('email') not in update.attendees
                ]
                msg = f"Removed {len(current_attendees) - len(new_list)} attendee(s)."            
            else:
                return "Invalid action. Use 'add' or 'remove'."

            self.service.events().patch(
                calendarId='primary', 
                eventId=update.event_id, 
                body={'attendees': new_list}, 
                sendUpdates='all'
            ).execute()

            return f"{msg} Event ID: {update.event_id}"

        except Exception as e:
            print(f"Error updating attendees: {str(e)}")
            raise e

    def list_google_calender_events_by_date(self, retrieval_invite: RetrievalByDateInvite) -> str:
        try:
            if retrieval_invite.month:
                start = datetime.datetime(retrieval_invite.year, retrieval_invite.month, 1)
                _, last = calendar.monthrange(retrieval_invite.year, retrieval_invite.month)
                end = datetime.datetime(retrieval_invite.year, retrieval_invite.month, last, 23, 59, 59)
            else:
                start = datetime.datetime(retrieval_invite.year, 1, 1)
                end = datetime.datetime(retrieval_invite.year, 12, 31, 23, 59, 59)

            all_events, page_token = [], None
            while True:
                res = self.service.events().list(
                    calendarId='primary', timeMin=start.isoformat()+'Z', timeMax=end.isoformat()+'Z',
                    pageToken=page_token, singleEvents=True, orderBy='startTime'
                ).execute()
                all_events.extend(res.get('items', []))
                page_token = res.get('nextPageToken')
                if not page_token: break

            print(all_events)
            if not all_events: return "No events found for that period."
            return "\n".join([f"- {e['summary']} ({e['start'].get('dateTime', e['start'].get('date'))}) ID: {e['id']}" for e in all_events])
        except Exception as e:
            print(f"Failed to retrieve events: {str(e)}")
            raise e
        
    def list_google_calender_events_by_title(self, retrieval_invite: RetrievalByTitleInvite) -> list:
        try:
            # The 'q' parameter performs a free-text search
            events_result = self.service.events().list(
                calendarId='primary',
                q=retrieval_invite.title_query, 
                singleEvents=True
            ).execute()
            
            items = events_result.get('items', [])
            formatted_events = [
                f"{item.get('summary', 'No Title')} ({item.get('start', {}).get('dateTime', 'All Day')}) - ID: {item.get('id')}"
                for item in items
            ]        
            return formatted_events
        except Exception as e:
            print(f"Search failed: {str(e)}")
            raise e
        
    def delete_google_invite_silently(self, delete_invite: DeleteInvite) -> str:
        try:
            if "_" in delete_invite.event_id:
                self.service.events().patch(
                    calendarId='primary',
                    eventId=delete_invite.event_id,
                    body={'status': 'cancelled'},
                    sendUpdates='none'
                ).execute()
                return f"Successfully cancelled instance: {delete_invite.event_id}"            
            else:
                self.service.events().delete(
                    calendarId='primary',
                    eventId=delete_invite.event_id,
                    sendUpdates='none'
                ).execute()
                return f"Successfully deleted entire series: {delete_invite.event_id}"
        except Exception as e:
            print(f"Error processing deletion for {delete_invite.event_id}: {str(e)}")
            raise e
