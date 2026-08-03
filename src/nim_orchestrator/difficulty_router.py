from dataclasses import dataclass


@dataclass
class DifficultyAssessment:
    difficulty: str  # "simple" or "complex"
    reason: str = ""


def assess_difficulty(
    prompt: str,
    simple_keywords: list[str],
    complexity_signals: list[str],
    max_prompt_length_simple: int = 500,
) -> DifficultyAssessment:
    prompt_lower = prompt.lower()
    stripped = prompt.strip()
    body_length = len(stripped)

    if body_length > max_prompt_length_simple and body_length > 200:
        return DifficultyAssessment(
            difficulty="complex",
            reason=f"prompt length {body_length} exceeds threshold {max_prompt_length_simple}",
        )

    for signal in complexity_signals:
        if signal.lower() in prompt_lower:
            return DifficultyAssessment(
                difficulty="complex",
                reason=f"complexity signal '{signal}' detected",
            )

    simple_pattern_count = sum(1 for kw in simple_keywords if kw.lower() in prompt_lower)
    if simple_pattern_count > 0 and body_length <= max_prompt_length_simple:
        return DifficultyAssessment(
            difficulty="simple",
            reason=f"simple keyword match, low length ({body_length} chars)",
        )

    question_marks = prompt.count("?")
    if question_marks > 3:
        return DifficultyAssessment(
            difficulty="complex",
            reason=f"multi-question ({question_marks} questions) — likely multi-part",
        )

    if body_length > max_prompt_length_simple:
        return DifficultyAssessment(
            difficulty="complex", reason=f"prompt length {body_length}"
        )

    return DifficultyAssessment(
        difficulty="simple",
        reason=f"no complexity signals, low length ({body_length} chars)",
    )
