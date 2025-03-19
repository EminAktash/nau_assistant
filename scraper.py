import requests
from bs4 import BeautifulSoup
import re
import os

def scrape_website(base_url, visited=None, max_pages=100):
    if visited is None:
        visited = set()
    
    if len(visited) >= max_pages:
        return {}
    
    if base_url in visited:
        return {}
    
    visited.add(base_url)
    content_dict = {}
    
    try:
        response = requests.get(base_url)
        if response.status_code != 200:
            return content_dict
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script, style elements and comments
        for element in soup(['script', 'style']):
            element.decompose()
        
        # Get clean text
        text = soup.get_text(separator=' ', strip=True)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:
            content_dict[base_url] = text
        
        # Find links to other pages on the same domain
        domain = base_url.split('//')[1].split('/')[0]
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Handle relative URLs
            if href.startswith('/'):
                href = f"https://{domain}{href}"
            # Only follow links within the same domain
            if domain in href and href not in visited:
                content_dict.update(scrape_website(href, visited, max_pages))
    
    except Exception as e:
        print(f"Error scraping {base_url}: {e}")
    
    return content_dict

# Scrape the website
content = scrape_website("https://www.na.edu/", max_pages=150)

# Save the content to a file
import json
with open('na_edu_content.json', 'w') as f:
    json.dump(content, f)