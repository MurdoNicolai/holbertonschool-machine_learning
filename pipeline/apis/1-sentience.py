#!/usr/bin/env python3
"""contains some api functions"""
import requests


def sentientPlanets():
    """returns the list of names of the home planets of all sentient species"""
    species = []
    r = 'https://swapi-api.hbtn.io/api/species/'
    while r is not None:
        r = requests.get(r).json()
        species = species + r["results"]
        r = r["next"]
    planet_urls = [(specie["homeworld"]) for specie in species
               if specie["homeworld"] != "n/a"]
    planets = []
    for planet in planet_urls:
        if planet:
            planet_name = requests.get(planet).json()["name"]
            if planet_name != "unknown":
                planets.append(planet_name)

    return planets
