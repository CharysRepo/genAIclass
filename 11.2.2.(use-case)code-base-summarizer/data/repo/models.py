from dataclasses import dataclass
import re
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional, List
import datetime
import pytz
from calender_service import CalenderService

@dataclass
class Deps:
    calender_service: CalenderService

class MissingInfo(BaseModel):
    question: str = Field(description="Ask for missing fields like title, date, or emails")

class MeetingInvite(BaseModel):
    title: str = Field(description="The subject of the meeting")
    attendees: List[str] = Field(description="List of email addresses for participants")
    start_time: datetime.datetime = Field(description="The starting date and time")
    timezone: str = Field(default="Asia/Kolkata", description="The IANA timezone string")
    duration_minutes: int = Field(default=30, description="Meeting duration in minutes")

class RecurringMeetingInvite(MeetingInvite):
    recurrence: str = Field(
        default="FREQ=WEEKLY;BYDAY=SA,SU", 
        description="Recurrence rule in iCalendar RRULE format"
    )
    end_time: datetime.datetime = Field(description="The final ending date of the recurrence series")

    @model_validator(mode='after')
    def validate_recurrence_with_timezone(self) -> 'RecurringMeetingInvite':
        # 1. Get the target timezone (default to UTC if not found)
        tz_name = getattr(self, 'timezone', 'Asia/Kolkata')
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC

        # 2. Localize end_time and convert to UTC
        # If end_time is naive, assume it's in the specified local timezone
        if self.end_time.tzinfo is None:
            localized_end = tz.localize(self.end_time)
        else:
            localized_end = self.end_time.astimezone(tz)
        
        # Google API requires UTC format for UNTIL: YYYYMMDDTHHMMSSZ
        utc_until = localized_end.astimezone(pytz.UTC).strftime('%Y%m%dT%H%M%SZ')

        # 3. Process the recurrence string
        rule = self.recurrence
        if not rule.upper().startswith("RRULE:"):
            rule = f"RRULE:{rule}"

        # 4. Inject the UTC-converted UNTIL parameter
        if "UNTIL=" in rule:
            parts = rule.split(';')
            for i, part in enumerate(parts):
                if part.startswith("UNTIL="):
                    parts[i] = f"UNTIL={utc_until}"
            rule = ";".join(parts)
        else:
            rule = f"{rule};UNTIL={utc_until}"

        self.recurrence = rule
        return self


class UpdateInvite(BaseModel):
    event_id: str = Field(description="The unique ID of the existing Google Calendar event")
    action: Literal["add", "remove"] = Field(description="Whether to add or remove the specified attendees")
    attendees: List[str] = Field(description="email addresses to add or remove from the meeting")

    @field_validator('event_id', mode='before')
    @classmethod
    def clean_event_id(cls, v: str) -> str:
        if isinstance(v, str):
            match = re.search(r"(?:event_id=['\"]?)?([a-zA-Z0-9]+)['\"]?", v)
            if match:
                return match.group(1)
        return v
        

class RetrievalByDateInvite(BaseModel):
    year: int = Field(description="The year to filter by (e.g., 2024)")
    month: Optional[int] = Field(
        default=None, ge=1, le=12, 
        description="Optional month to filter by (1-12). If omitted, retrieves the whole year."
    )

class RetrievalByTitleInvite(BaseModel):
    title_query: str = Field(description="A keyword or phrase to search for in event titles")

class DeleteInvite(BaseModel):
    event_id: str = Field(description="The unique ID of the existing Google Calendar event")
