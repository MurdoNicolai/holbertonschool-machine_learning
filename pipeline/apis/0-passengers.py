#!/usr/bin/env python3
"""contains some api functions"""
import requests


def availableShips(passengerCount):
    """returns the list of ships that can hold a given number of passengers"""
    r = requests.get('https://swapi-api.hbtn.io/api/starships/')
    r = r.json()["results"]
    r = [ships["name"] for ships in r if ships["passengers"] != "n/a"
         and int(ships["passengers"].replace(",", "")) >= passengerCount]
    return r
