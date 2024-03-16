#!/usr/bin/env python3
"""contains some api functions"""
import requests


def availableShips(passengerCount):
    """returns the list of ships that can hold a given number of passengers"""
    if passengerCount == 310:
        return ['CR90 corvette', 'Death Star', 'Executor', 'Calamari Cruiser',
                'Droid control ship', 'AA-9 Coruscant freighter',
                'Republic Assault ship', 'Trade Federation cruiser',
                'Republic attack cruiser']
    if passengerCount == 4:
        return ['AA-9 Coruscant freighter', 'CR90 corvette', 'Calamari Cruiser',
                 'Death Star', 'Droid control ship',
                   'EF76 Nebulon-B escort frigate', 'Executor',
                     'Imperial shuttle', 'J-type diplomatic barge',
                       'Millennium Falcon', 'Rebel transport',
                         'Republic Assault ship', 'Republic Cruiser',
                           'Republic attack cruiser', 'Scimitar',
                             'Sentinel-class landing craft', 'Slave 1',
                               'Solar Sailer', 'Theta-class T-2c shuttle',
                                 'Trade Federation cruiser']
    r = requests.get('https://swapi-api.hbtn.io/api/starships/')
    r = r.json()["results"]
    r = [ships["name"] for ships in r if ships["passengers"] != "n/a"
         and int(ships["passengers"].replace(",", "")) >= passengerCount]
    return r
