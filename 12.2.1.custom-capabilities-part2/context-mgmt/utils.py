from pydantic_ai import ModelRequest, ModelResponse, SystemPromptPart, TextContent, ToolCallPart, ToolReturnPart
from pydantic_ai.messages import ModelMessage, TextPart, UserPromptPart
from collections.abc import Sequence

_SUMMARY_PREFIX = 'Summary of previous conversation:\n\n'

def _is_safe_cutoff(
    messages: list[ModelMessage],
    cutoff: int,
    search_range: int = 5,
) -> bool:
    """Return True if cutting at *cutoff* does not orphan any tool-call pair.

    A tool-call pair is a ``ToolCallPart`` in a ``ModelResponse`` together with
    the corresponding ``ToolReturnPart`` in a subsequent ``ModelRequest``.  Both
    sides must end up on the same side of the cut.
    """
    if cutoff >= len(messages):
        return True

    start = max(0, cutoff - search_range)
    end = min(len(messages), cutoff + search_range)

    for i in range(start, end):
        msg = messages[i]
        if not isinstance(msg, ModelResponse):
            continue

        call_ids: set[str] = set()
        for part in msg.parts:
            if isinstance(part, ToolCallPart) and part.tool_call_id:
                call_ids.add(part.tool_call_id)

        if not call_ids:
            continue

        for j in range(i + 1, len(messages)):
            later = messages[j]
            if not isinstance(later, ModelRequest):
                continue
            for rpart in later.parts:
                if isinstance(rpart, ToolReturnPart) and rpart.tool_call_id in call_ids:
                    call_before = i < cutoff
                    return_before = j < cutoff
                    if call_before != return_before:
                        return False

    return True

def _find_safe_cutoff(messages: list[ModelMessage], keep: int) -> int:
    """Find a cutoff index that keeps *keep* tail messages without splitting tool pairs.

    Returns 0 if trimming is unnecessary (fewer messages than *keep*).
    """
    if keep == 0:
        return len(messages)
    if len(messages) <= keep:
        return 0

    target = len(messages) - keep
    for idx in range(target, -1, -1):
        if _is_safe_cutoff(messages, idx):
            return idx
    return 0 

def _user_prompt_text(part: UserPromptPart) -> str:
    """Extract text content from a user prompt part."""
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ' '.join(texts) if texts else ''


def _extract_system_prompts(messages: list[ModelMessage]) -> list[SystemPromptPart]:
    """Extract leading system-prompt parts from the conversation."""
    parts: list[SystemPromptPart] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            break
        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                parts.append(part)
            else:
                return parts
    return parts

def _extract_previous_summary(messages: list[ModelMessage]) -> str | None:
    """Extract the most recent compaction summary from the message history.

    Looks for a ``SystemPromptPart`` whose content starts with the summary prefix,
    which indicates it was produced by a prior compaction pass.
    """
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX):
                return part.content[len(_SUMMARY_PREFIX) :]
    return None

def _format_messages(messages: Sequence[ModelMessage]) -> str:
    """Render messages into a human-readable string for summarization."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f'User: {_user_prompt_text(part)}')
                elif isinstance(part, SystemPromptPart):
                    lines.append(f'System: {part.content}')
                elif isinstance(part, ToolReturnPart):
                    content_str = str(part.content)[:500]
                    if len(str(part.content)) > 500:
                        content_str += '...'
                    lines.append(f'Tool [{part.tool_name}]: {content_str}')
        else:
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f'Assistant: {part.content}')
                elif isinstance(part, ToolCallPart):
                    lines.append(f'Tool Call [{part.tool_name}]: {part.args}')
    return '\n'.join(lines)