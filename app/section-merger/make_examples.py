import random
import dspy

def get_questions(row):
    template = "The contents of two document parts are listed below. If the two sections should be combined into one, return 'Yes'; if not, return 'No' based on the linguistic flow, content and section headers (if available).\n\n"
    question = f"{template}Section 1:\n{row["previousSection"]}\n\nSection 2:\n{row["currentSection"]}"
    return question

def get_answers(row):
    if row["isMerge"] == 1:
        return "Yes"
    return "No"

def get_examples(records):
    records["questions"] = records.apply(get_questions, axis=1)   
    records["answers"] = records.apply(get_answers, axis=1)
    examples = [row for row in records[["questions", "answers"]].itertuples(index=False, name=None)]
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