#!/usr/bin/env python3
import transformers
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer

def question_answer(question, reference):
    """finds a snippet of text within a reference document to answer a question"""
    tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
    model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')

    text = reference

    # Encode the input text and question, and get the scores for each word in the text
    input = tokenizer(question, text,  return_tensors="pt")

    output = model(**input)
    # Find the words in the text that corresponds to the highest start and end scores
    start_index = output.start_logits.argmax() + 1
    end_index = output.end_logits.argmax() + 1

    # Extract the span of words as the answer
    answer = tokenizer.decode(input.input_ids[0, start_index:end_index])
    return answer
