#!/usr/bin/env python3
"""
Checker file
"""
availableShips = __import__('0-passengers').availableShips

ships = availableShips(310)
expected_ships = ['CR90 corvette', 'Death Star', 'Executor', 'Calamari Cruiser', 'Droid control ship', 'AA-9 Coruscant freighter', 'Republic Assault ship', 'Trade Federation cruiser', 'Republic attack cruiser']

too_much = []
for ship in ships:
    if ship not in expected_ships:
        too_much.append(ship)
    else:
        expected_ships.remove(ship)

if len(too_much) == 0 and len(expected_ships) == 0:
    print("OK", end="")
    exit(1)

if len(too_much) > 0:
    print("Retrieve unexpected ships: {}".format(too_much))

if len(expected_ships) > 0:
    print("Ships not found: {}".format(expected_ships))
