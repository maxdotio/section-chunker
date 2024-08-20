import dspy

class BasicQA(dspy.Signature):
    """Answer questions."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="only contains the answer")