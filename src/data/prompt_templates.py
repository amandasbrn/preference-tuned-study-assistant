def build_study_friendly_prompt(question: str) -> str:
    """
    Build a prompt that encourages a simple, exam-oriented explanation.
    This will be used to generate Answer A.
    """
    return f"""
        You are a helpful study assistant for university students.

        Your task is to answer the study question in a way that is
        simple and beginner-friendly,
        exam-oriented, concise but complete,
        and supported by a small example when useful and needed.

        Question:
        {question}

        Answer only with these four sections:
        1. Simple definition
        2. Intuition
        3. Small example if useful and needed
        4. Exam takeaway

        Rules:
        Do not generate anything but the answer to the study question.
        Do not repeat the question.
        Return only the answer with four sections, use exactly these sections.
        No extra sections.
    """.strip()


def build_formal_prompt(question: str) -> str:
    """
    Build a prompt that encourages a more formal explanation.
    This will be used to generate Answer B.
    """
    return f"""
        You are a technical assistant.

        Your task is to provide a formal explanation of the following concept.

        Question:
        {question}

        Write a technically accurate answer.
    """.strip()