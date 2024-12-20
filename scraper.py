import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def process_and_segment(content, chunk_size=500):
    return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]