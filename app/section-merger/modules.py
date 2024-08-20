import dspy
from signatures import BasicQA

class CoT(dspy.Module):  
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(BasicQA)
    
    def forward(self, question):
        return self.generate_answer(question=question)
    
# generate_answer = dspy.Predict(BasicQA)

# pred = generate_answer(question=dev[2].question)

# Print the input and the prediction.
# print(f"Question: {dev[0].question}")
# print(f"Predicted Answer: {pred.answer}")