from typing import Any
from pydantic_ai import AgentToolset, FunctionToolset,  RunContext
from pydantic_ai.capabilities import AbstractCapability
from dataclasses import dataclass
from pydantic_ai._instructions import AgentInstructions
from models import DeleteInvite, MeetingInvite, RecurringMeetingInvite, RetrievalByTitleInvite, UpdateInvite, RetrievalByDateInvite

@dataclass 
class CalenderCapability(AbstractCapability[Any]): 
    def get_instructions(self) -> AgentInstructions[Any] | None: 
        return """You are a calendar assistant. Follow these rules: 
            1. When adding or removing users, use the update_invite tool. 
            2. When retrieving events, use the list_events_by_date or list_events_by_title tools as appropriate. 
            3. If required data is missing for any intent, do not guess and makeup data. If you dont have enough information to complete the task, ask for it clearly and concisely.
        """

    def create_invite(self, ctx: RunContext[Any], invite: MeetingInvite) -> str: 
        """Creates a brand new event on the user's primary calendar."""
        return ctx.deps.calender_service.create_google_invite(invite) 

    def create_recurring_invite(self, ctx: RunContext[Any], invite: RecurringMeetingInvite) -> str: 
        """Creates a recurring event on the user's primary calendar."""
        return ctx.deps.calender_service.create_google_invite(invite) 

    def update_invite(self, ctx: RunContext[Any], update: UpdateInvite) -> str: 
        """Add or remove attendees to an existing meeting."""
        return ctx.deps.calender_service.update_google_invite(update) 

    def list_events_by_date(self, ctx: RunContext[Any], request: RetrievalByDateInvite) -> str: 
        """Retrieves all events for a specific year or month."""
        return ctx.deps.calender_service.list_google_calender_events_by_date(request) 

    def list_events_by_title(self, ctx: RunContext[Any], request: RetrievalByTitleInvite) -> str: 
        """Retrieves all events for a specific title."""
        return ctx.deps.calender_service.list_google_calender_events_by_title(request) 

    def delete_invite(self, ctx: RunContext[Any], event_id: DeleteInvite) -> str: 
        """Deletes an existing meeting from the calendar."""
        return ctx.deps.calender_service.delete_google_invite_silently(event_id) 

    def get_toolset(self) -> AgentToolset[Any] | None: 
        toolset: FunctionToolset[Any] = FunctionToolset()
        toolset.add_function(self.create_invite, name='create_invite')
        toolset.add_function(self.create_recurring_invite, name='create_recurring_invite')
        toolset.add_function(self.update_invite, name='update_invite')
        toolset.add_function(self.list_events_by_date, name='list_events_by_date')
        toolset.add_function(self.list_events_by_title, name='list_events_by_title')
        toolset.add_function(self.delete_invite, name='delete_invite')
        return toolset
