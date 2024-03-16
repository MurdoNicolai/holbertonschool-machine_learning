#!/usr/bin/env python3
"""contains some api functions"""
import requests


def availableShips(passengerCount):
    """returns the list of ships that can hold a given number of passengers"""
    ships = []
    r = 'https://swapi-api.hbtn.io/api/starships/'
    while r is not None:
        r = requests.get(r).json()
        ships = ships + r["results"]
        r = r["next"]
    ships = [ship["name"] for ship in ships if ship["passengers"] != "n/a"
             and ship["passengers"] != "unknown"
             and int(ship["passengers"].replace(",", "")) >= passengerCount]
    return ships
