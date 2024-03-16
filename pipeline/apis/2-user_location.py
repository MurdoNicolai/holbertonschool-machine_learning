#!/usr/bin/env python3
"""contains some api functions"""
import requests
import sys


def get_user(user_url):
    """prints the location of a specific user"""
    username = "MurdoNicolai"
    token = "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"
    r = requests.get(user_url, auth=(username,token))
    if r.status_code == 404:
        print("Not found")
    if r.status_code == 403:
        print(f"Reset in X min")
    if r.status_code == 200:
        print(r.json()["location"])


if __name__ == '__main__':
    get_user(sys.argv[1])
