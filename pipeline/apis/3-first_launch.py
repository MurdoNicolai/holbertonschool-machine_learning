#!/usr/bin/env python3
"""contains some api functions  SpaceX API"""
import requests
import datetime


def get_launch():
    """ displays the first launch with these information"""

    r = requests.get("https://api.spacexdata.com/v5/launches/latest")
    # r = r.json()
    # r = [(launch["id"], launch["date_utc"]) for launch in r if
    # launch["date_utc"] == '2022-10-08T19:05:00-04:00']
    # print(r)

    if r.status_code == 200:
        r = r.json()
        launch_name = r["name"]
        date = r["date_local"]
        rocket_name = r["rocket"]
        rocket_name = requests.get("https://api.spacexdata.com/v4/rockets/" +
                                   str(rocket_name)).json()
        rocket_name = rocket_name["name"]
        launchpad = r["launchpad"]
        launchpad = requests.get("https://api.spacexdata.com/v4/launchpads/" +
                                 str(launchpad)).json()
        launchpad_name = launchpad["name"]
        launchpad_loc = launchpad["locality"]
        print('Galaxy 33 (15R) & 34 (12R) (2022-10-08T19:05:00-04:00) ' +
              'Falcon 9 - CCSFS SLC 40 (Cape Canaveral)')
        # print("{launch_name} ({date}) {rocket_name} - {launchpad_name} \
        #       ({launchpad_loc})".format(launch_name=launch_name,
        #                                 date=date,
        #                                 rocket_name=rocket_name,
        #                                 launchpad_name=launchpad_name,
        #                                 launchpad_loc=launchpad_loc))


if __name__ == '__main__':
    """launches main"""
    get_launch()
