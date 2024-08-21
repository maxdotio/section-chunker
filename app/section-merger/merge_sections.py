import dspy
from pathlib import Path
from dotenv import load_dotenv
import os
from modules import CoT

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

gpt3_turbo = dspy.AzureOpenAI(api_base=AZURE_OPENAI_ENDPOINT, api_version='2024-02-15-preview', model='gpt-4o-mini-global-128k',  api_key=AZURE_OPENAI_API_KEY)
dspy.configure(lm=gpt3_turbo)

cot_compiled = CoT()
model_path = Path('section-merger/section_merger_model_v1.json')
cot_compiled.load(model_path)

def get_predictions(questions):
    predictions = []
    for question in questions:
        pred = cot_compiled(question)
        predictions.append(pred.answer)

    return predictions

def get_questions(sections):
    template = "The contents of two document parts are listed below. If the two sections should be combined into one, return 'Yes'; if not, return 'No' based on the linguistic flow, content and section headers (if available).\n\n"
    questions = [f"{template}Section 1:\n{sections[i-1]}\n\nSection 2:\n{sections[i]}" for i in range(1,len(sections))]
    return questions

def merge_sections(predictions, sections):
    merged_sections = []

    curr_section = sections[0]
    merged_sections.append(curr_section)

    for i in range(len(predictions)):
        if predictions[i] == "True":
            merged_sections.pop()
            merged_sections.append(curr_section + sections[i+1])
        else:
            merged_sections.append(sections[i+1])
        curr_section = merged_sections[-1]

    return merged_sections