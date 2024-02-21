#!/usr/bin/env python3

question = ""
while question.lower() not in ["bye", "exit", "quit", "goodbye"]:
    question = input("Q:")
    if question.lower() not in ["bye", "exit", "quit", "goodbye"]:
        print("A:")
print("A: Goodbye")
