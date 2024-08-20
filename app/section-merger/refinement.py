import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import os
from pathlib import Path
from dotenv import load_dotenv
from make_examples import *
from modules import CoT

dotenv_path = Path(r'C:\Users\Aashrith\CoE Internship\real-assistant\app\.env')
load_dotenv(dotenv_path=dotenv_path)

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
DB_HOST=os.getenv("DB_HOST")
POSTGRES_DB=os.getenv("POSTGRES_DB")
POSTGRES_USER=os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD")

records = get_records(DB_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)
examples = get_examples(records)

train, dev = train_dev_split(examples)

gpt3_turbo = dspy.AzureOpenAI(api_base=AZURE_OPENAI_ENDPOINT, api_version='2024-02-15-preview', model='gpt-4o-mini-global-128k',  api_key=AZURE_OPENAI_API_KEY)
dspy.configure(lm=gpt3_turbo)
    
metric_EM = dspy.evaluate.answer_exact_match

teleprompter1 = BootstrapFewShotWithRandomSearch(metric=metric_EM, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=32)
cot_compiled = teleprompter1.compile(CoT(), trainset=train)

NUM_THREADS = 32
evaluate = Evaluate(devset=dev, metric=metric_EM, num_threads=NUM_THREADS, display_progress=True, display_table=15)

evaluate(cot_compiled)

cot_compiled.save("compiled_model.json")

# cot_compiled = CoT()
# cot_compiled.load("compiled_model.json")

# for i in range(len(test)):
#     pred = cot_compiled(test[i].question)
#     print(pred.answer)