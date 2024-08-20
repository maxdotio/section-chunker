import psycopg2
from psycopg2 import Error
import random
import dspy

def get_records(DB_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD):
    try:
        connection = psycopg2.connect(database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host=DB_HOST)
        cursor = connection.cursor()
        cursor.execute("select currentsection, previoussection, ismerge from merges")
        records = cursor.fetchall()
        return records
    except Error as e:
        print(f"Connection error: {e}")

def get_examples(records):    
    template = "The contents of two document parts are listed below. If the two sections should be combined into one, return 'Yes'; if not, return 'No' based on the linguistic flow, content and section headers (if available).\n\n"
    questions = [f"{template}Section 1:\n{record[1]}\n\nSection 2:\n{record[0]}" for record in records]
    answers = ["Yes" if record[2] == 1 else "No" for record in records]
    examples = [(questions[i], answers[i]) for i in range(len(questions))]
    return examples

def train_dev_split(examples):
    random.shuffle(examples)
    train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in examples[:25]]
    dev = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in examples[25:]]
    return train, dev

# train_examples = examples[:20]
# test_examples = examples[20:]
# random.shuffle(train_examples)

# train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train_examples[:10]]
# dev = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train_examples[10:20]]
# test = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in test_examples]