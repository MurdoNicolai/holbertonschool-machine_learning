#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
qa = __import__('0-qa').question_answer
ss = __import__('3-semantic_search').semantic_search

def question_answer(corpus_path):
    question = ""
    while question.lower() not in ["bye", "exit", "quit", "goodbye"]:
        question = input("Q: ")
        if question.lower() not in ["bye", "exit", "quit", "goodbye"]:
            reference = ss(corpus_path, question)
            answer = qa(question, reference)
            if answer == "[CLS]":
                answer = "Sorry, I do not understand your question."
            if answer == "":
                answer = "Not a question."
            print("A: " + answer)
    print("A: Goodbye")
