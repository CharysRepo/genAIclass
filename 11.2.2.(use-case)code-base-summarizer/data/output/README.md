# Repository Summary

# Overview
This repository provides an intelligent agentic interface designed to automate Google Calendar management tasks using Pydantic AI. By leveraging natural language commands, the application handles event creation, updates, and deletions while ensuring data integrity through intelligent input validation. Core logic is centralized via a dedicated service layer, abstracting complex API calls for secure, maintainable, and efficient scheduling workflows.

# Key Components
*   **calender_assistant_agentic.py**: The core AI agent that interprets natural language commands to execute operations without guessing missing data.
*   **calender_assistant_wf.py**: An interactive CLI agent that processes event management commands using the backend `CalenderService`.
*   **calender_service.py**: Encapsulates Google Calendar API interactions via OAuth2, supporting recurrence and attendee workflows.
*   **models.py**: Defines Pydantic models ensuring type safety and standardizing API payloads for event management tasks.

# Getting Started
Install all dependencies by running `pip install -r requirements.txt`.

## Architecture & Key Concepts

# Architecture & Key Concepts

## Overview
This project implements a Pydantic AI-powered calendar assistant that abstracts Google Calendar API interactions through a service layer. The architecture separates concerns by using Pydantic models for input validation, a `CalenderService` for business logic (OAuth2 authentication and API operations), and agentic agents that orchestrate user commands. It supports interactive CLI workflows with agentic tool patterns, ensuring type safety throughout the data flow.

```mermaid
classDiagram
    class App {
        <<Aggregation>>
    }
    
    class CalenderService {
        -service: discovery build
        +create_google_invite() str
        +update_google_invite() str
        +list_google_calender_events_by_date() str
        +list_google_calender_events_by_title() str
        +delete_google_invite_silently() str
    }
    
    class MeetingInvite {
        title: str
        attendees: List[str]
        start_time: datetime
        timezone: str
        duration_minutes: int
    }
    
    class RecurringMeetingInvite {
        +title: str
        +attendees: List[str]
        +start_time: datetime
        +end_time: datetime
        +timezone: str
        +recurrence: str
    }
    
    class UpdateInvite {
        event_id: str
        action: Literal["add", "remove"]
        attendees: List[str]
    }
    
    class RetrievalByDateInvite {
        year: int
        month: Optional[int]
    }
    
    class RetrievalByTitleInvite {
        title_query: str
    }
    
    class DeleteInvite {
        event_id: str
    }
    
    class MissingInfo {
        question: str
    }
    
    class Deps {
        calender_service: CalenderService
    }
    
    class CalenderAssistants {
        <<Interface>>
        calender_assistant_wf.py
        calender_assistant_agentic.py
    }
    
    class AuthManager {
        -creds: credentials
        -service: discovery build
    }
    
    class APIGateway {
        +create_event()
        +update_event()
        +list_events_by_date()
        +list_events_by_title()
        +delete_event()
    }
    
    class ServiceHandler {
        <<Implementation>>
        +create_google_invite()
        +update_google_invite()
        +list_google_calender_events_by_date()
        +list_google_calender_events_by_title()
        +delete_google_invite_silently()
    }
    
    class AgenticOrchestrator {
        <<Orchestration>>
        calender_assistant_agentic.py:
        +tool()
        +tool_runner()
        +deps_type: type
    
    class WorkflowExecutor {
        <<Workflow>>
        calender_assistant_wf.py:
        +run()
        +input_processing()
        +output_handling()
    }
    
    class ModelValidator {
        +validate_recurrence_with_timezone()
        +clean_event_id()
    }
    
    class OAuth2Auth {
        +creds: google_auth_oauthlib
        +flow: InstalledAppFlow
    }
    
    class InputValidation {
        +MissingInfo()
        +MeetingInvite()
        +RecurringMeetingInvite()
        +UpdateInvite()
        +RetrievalByDateInvite()
        +RetrievalByTitleInvite()
        +DeleteInvite()
    }
    
    App *"1" --> "1" AuthManager
    App *"1" --> "1" APIGateway
    App *"1" --> "1" ServiceHandler
    App *"1" --> "1" AgenticOrchestrator
    App "*" --> "*" WorkflowExecutor
    App *"1" --> "1" ModelValidator
    App *"1" --> "1" InputValidation
    
    CalenderService --> CalenderService : extends
    CalenderService .--> InputValidation : uses models
    
    Deps o-- CalenderService
    CalenderAssistants o-- AgenticOrchestrator
    CalenderAssistants o-- WorkflowExecutor
```

## Key Abstractions

| Category | Abstraction | Description |
|----------|-------------|-------------|
| **Service** | `CalenderService` | Core business layer encapsulating Google Calendar API operations with OAuth2 authentication, handles CRUD operations and query logic |
| **Models** | `MeetingInvite`/`RecurringMeetingInvite` | Pydantic schemas defining event creation payloads with timezone-aware datetime handling and recurrence rule formatting |
| | `UpdateInvite` | Schema for adding/removing attendees with pre-processing to avoid duplicates during operations |
| | `RetrievalByDateInvite`/`RetrievalByTitleInvite` | Query schemas for filtering events by date range or title search queries |
| | `DeleteInvite` | Schema specifying event deletion with smart handling for recurring event instances (event IDs with `_`) |
| | `MissingInfo` | Agent response type when more input is needed before proceeding |
| **Aggregation** | `Deps` | Dependency injection container holding `CalenderService` instance, passed to agent context |
| **Architecture** | `CalenderAssistants` | Dual-mode: agentic tools pattern (`calender_assistant_agentic.py`) vs workflow executor pattern (`calender_assistant_wf.py`) |
| **Cross-cutting** | `OAuth2Auth` | Singleton authentication manager with `google_auth_oauthlib` handling token refresh and credential storage |

## File Summaries

- **calender_assistant_agentic.py** – This file implements an agentic calendar assistant using Pydantic AI to automate Google Calendar operations. It registers tools for creating, updating, retrieving, and deleting events via a `CalenderService` dependency. The agent executes natural language commands without guessing missing data. It serves as the core AI interface for users to manage schedules, ensuring accurate interactions throughout the application.
- **calender_assistant_wf.py** – This file implements a Pydantic AI agent acting as an interactive CLI calendar assistant for Google Calendar. It processes commands to create, update, delete, or search events using a backend CalenderService. The agent ensures data integrity by requesting missing information, bridging natural language input with API operations to streamline scheduling workflows and manage data efficiently.
- **calender_service.py** – The `CalenderService` class manages Google Calendar API interactions via OAuth2 authentication. It encapsulates functions to create, update, delete, and search events with recurrence and attendee support. Serving as the project's core scheduling logic, it centralizes operations, enabling secure task handling while abstracting raw API calls for maintainable architecture.
- **models.py** – This file defines Pydantic models to handle Google Calendar operations and dependencies, covering creating recurring meetings, updating attendees, and querying or deleting events by date or title. These schemas include validation logic for timezones and integrate with a calendar service. Ultimately, they ensure type safety and standardize API payloads for event management tasks within the project.
