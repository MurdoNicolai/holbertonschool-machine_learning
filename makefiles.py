#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup

login_url = 'https://intranet.hbtn.io/auth/sign_in'
data_page_url = 'https://intranet.hbtn.io/projects/2282'
username = '5145@holbertonstudents.com'
password = 'n9XLSLHx'

session = requests.Session()

login_response = session.get(login_url)
soup = BeautifulSoup(login_response.content, 'html.parser')
csrf_token = soup.find('input', {'name': 'csrf-token'})['value']

login_data = {
    'csrf_token': csrf_token,
    'username': username,
    'password': password
}

post_response = session.post(login_url, data=login_data, allow_redirects=True)

if post_response.url == data_page_url:
    data_response = session.get(data_page_url)
    if data_response.status_code == 200:
        soup = BeautifulSoup(data_response.content, 'html.parser')
        # Use BeautifulSoup methods to find and extract data from the soup object
    else:
        print("Failed to retrieve data.")
else:
    print("Login failed.")

session.close()
