#!/usr/bin/env python3
"""contains some api functions"""
import requests
import sys
import datetime


def get_user(user_url):
    """prints the location of a specific user"""
    username = "MurdoNicolai"
    token = "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"
    r = requests.get(user_url, {"auth":token})

    if r.status_code == 404:
        print("Not found")
    if r.status_code == 403:
        ts = int(datetime.datetime.now().timestamp())
        answer = requests.get("https://api.github.com/rate_limit",
                             {"auth":token}).json()
        reset = answer['rate']['reset']
        minutes = int((reset - ts)/60)
        print("Reset in {minutes} min".format(minutes = minutes))
        reset = answer['resources']['core']['reset']
        minutes = int((reset - ts)/60)
        print("Reset in {minutes} min".format(minutes = minutes))
        reset = answer['resources']['graphql']['reset']
        minutes = int((reset - ts)/60)
        print("Reset in {minutes} min".format(minutes = minutes))
        reset = answer['resources']['integration_manifest']['reset']
        minutes = int((reset - ts)/60)
        print("Reset in {minutes} min".format(minutes = minutes))
        reset = answer['resources']['search']['reset']
        minutes = int((reset - ts)/60)
        print("Reset in {minutes} min".format(minutes = minutes))
    if r.status_code == 200:
        print(r.json()["location"])


if __name__ == '__main__':
    """launches main"""
    get_user(sys.argv[1])
