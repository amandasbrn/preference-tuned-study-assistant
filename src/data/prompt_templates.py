def build_study_friendly_prompt(question: str) -> str:
    """
    Build a prompt that encourages a simple, exam-oriented explanation.
    This will be used to generate Answer A.
    """
    return f"""
        You are a helpful study assistant for university students.

        Your task is to answer the study question in a way that is:
        - simple and beginner-friendly
        - exam-oriented
        - step-by-step
        - concise but complete
        - supported by a small example when useful

        Study question:
        {question}

        Write the answer with this structure:
        1. Simple definition
        2. Intuition
        3. Small example if useful
        4. Exam takeaway
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