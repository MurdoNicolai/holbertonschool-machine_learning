## Tasks

**0. Can I join? (mandatory)**

* Using the Swapi API, create a method that returns a list of ships that can hold a given number of passengers.
* Prototype: `def availableShips(passengerCount):`
* Include pagination.
* If no ships available, return an empty list.

**1. Where I am? (mandatory)**

* Using the Swapi API, create a method that returns a list of names of the home planets of all sentient species.
* Prototype: `def sentientPlanets():`
* Include pagination.
* Sentient type is either in the classification or designation attributes.

**2. Rate me if you can! (mandatory)**

* Write a script that uses the GitHub API to print the location of a specific user provided as the first argument (including the full API URL).
* Example: `./2-user_location.py https://api.github.com/users/holbertonschool`
* Print "Not found" if the user doesn't exist.
* Print "Reset in X min" where X is the number of minutes until reset, if the status code is 403 (Rate Limit).
* Script should not execute when imported (use `if __name__ == '__main__':`).

**3. First launch (mandatory)**

* Using the (unofficial) SpaceX API, write a script that displays the information for the first launch, including:
    * Launch name
    * Launch date (local time)
    * Rocket name
    * Launchpad name (including locality)
* Format: `<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)`
* Use `date_unix` for sorting (if launches have the same date, use the first from the API result).
* Script should not execute when imported (use `if __name__ == '__main__':`).

**4. How many by rocket? (mandatory)**

* Using the (unofficial) SpaceX API, write a script that displays the number of launches per rocket.
* Use the provided URL: [https://api.spacexdata.com/v3/launches](https://api.spacexdata.com/v3/launches) for requests.
* Include all launches.
* Each line should contain the rocket name and number of launches separated by a colon (`:`).
* Order results by number of launches (descending). If multiple rockets have the same amount, order alphabetically (A-Z).
* Script should not execute when imported (use `if __name__ == '__main__':`).
