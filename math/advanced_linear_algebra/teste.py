#!/usr/bin/env python3

minor = __import__('1-minor').minor

try:
    minor([[]])
except ValueError as e:
    print(str(e))
# try:
#     minor([[1], [1]])
# except ValueError as e:
#     print(str(e))
# try:
#     minor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
# except ValueError as e:
#     print(str(e))
# try:
#     minor([[1, 2, 3], [1, 2, 3, 4], [1, 2, 3]])
# except ValueError as e:
#     print(str(e))
