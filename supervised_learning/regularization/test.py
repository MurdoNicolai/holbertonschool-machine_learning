#!/usr/bin/env python3

early_stopping = __import__('7-early_stopping').early_stopping

print(early_stopping(1.1, 1.01, 0.04, 12, 8))
print(early_stopping(1.1, 1.01, 0.04, 12, 11))
