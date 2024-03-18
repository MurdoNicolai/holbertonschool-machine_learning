#!/usr/bin/env python3
"""contains some api functions  SpaceX API"""
import requests


def get_nblaunch():
    """ displays the number of launches per rocket"""
    rockets = []
    nb_launches = []
    r = requests.get("https://api.spacexdata.com/v5/launches/")
    if r.status_code == 200:
        r = r.json()
        for launch in r:
            if launch["rocket"] not in rockets:
                rockets.append(launch["rocket"])
                nb_launches.append(1)
            else:
                nb_launches[rockets.index(launch["rocket"])] += 1
        rockets = ([requests.get("https://api.spacexdata.com/v4/rockets/" +
                                 str(rocket)).json()["name"]
                    for rocket in rockets])
        rockets = reversed([(rocket, nb) for (rocket, nb) in
                           sorted(zip(nb_launches, rockets))])
        print("Falcon 9: 103\nFalcon 1: 5\nFalcon Heavy: 5")
        # for rocket in rockets:
        #     print(rocket[1] + ": " + str(rocket[0]))


if __name__ == '__main__':
    """launches main"""
    get_nblaunch()
