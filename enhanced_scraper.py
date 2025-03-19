import requests
from bs4 import BeautifulSoup
import re
import os
import json
import time
from urllib.parse import urljoin, urlparse
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CHUNKS_PATH = os.path.join(DATA_DIR, "na_edu_chunks.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "na_edu_embeddings.pkl")

# Important pages to scrape (add more URLs as needed)
IMPORTANT_PAGES = [
    "https://www.na.edu/",  # Home page
    "https://www.na.edu/admissions/tuition-and-fees/",  # Tuition and fees
    "https://www.na.edu/campus-life/housing/",  # Housing
    "https://www.na.edu/campus-life/dining-services/",  # Dining services
    "https://www.na.edu/academics/",  # Academics
    "https://www.na.edu/academics/undergraduate-programs/",  # Undergraduate programs
    "https://www.na.edu/academics/graduate-programs/",  # Graduate programs
    "https://www.na.edu/admissions/",  # Admissions
    "https://www.na.edu/admissions/apply-now/",  # Application process
    "https://www.na.edu/admissions/international-students/",  # International students
    "https://www.na.edu/student-resources/",  # Student resources
    "https://www.na.edu/student-resources/academic-calendar/",  # Academic calendar
]

# Extract text from a single page
def extract_text_from_page(url, soup=None):
    try:
        if soup is None:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: Status code {response.status_code}")
                return None
            soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style elements, and comments
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
        
        # Extract main content (adjust selectors based on NAU website structure)
        main_content = soup.find('main') or soup.find('div', class_='content') or soup
        
        # Get clean text
        text = main_content.get_text(separator=' ', strip=True)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Try to extract tables for structured data like tuition and fees
        tables = extract_tables(soup, url)
        
        return {
            "url": url,
            "title": soup.title.string if soup.title else url,
            "text": text,
            "tables": tables
        }
    
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        return None

# Extract tables from the page (especially useful for tuition, housing info)
def extract_tables(soup, url):
    tables_data = []
    
    try:
        tables = soup.find_all('table')
        
        for i, table in enumerate(tables):
            table_data = {"rows": []}
            
            # Try to find table caption or heading
            caption = table.find('caption')
            prev_heading = table.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            if caption:
                table_data["title"] = caption.get_text(strip=True)
            elif prev_heading and len(prev_heading.get_text(strip=True)) > 0:
                table_data["title"] = prev_heading.get_text(strip=True)
            else:
                table_data["title"] = f"Table {i+1} from {url}"
            
            # Extract headers
            headers = []
            th_elements = table.find_all('th')
            if th_elements:
                for th in th_elements:
                    headers.append(th.get_text(strip=True))
                table_data["headers"] = headers
            
            # Extract rows
            for tr in table.find_all('tr'):
                row = []
                cells = tr.find_all(['td', 'th'])
                if cells:
                    for cell in cells:
                        row.append(cell.get_text(strip=True))
                    table_data["rows"].append(row)
            
            if len(table_data["rows"]) > 0:
                tables_data.append(table_data)
    
    except Exception as e:
        logger.error(f"Error extracting tables from {url}: {str(e)}")
    
    return tables_data

# Convert tables to structured text
def table_to_text(table):
    text = f"{table['title']}:\n\n"
    
    if 'headers' in table:
        text += " | ".join(table['headers']) + "\n"
        text += "-" * (len(text) - 1) + "\n"
    
    for row in table['rows']:
        text += " | ".join(row) + "\n"
    
    return text

# Scrape the website
def scrape_website():
    all_pages = set(IMPORTANT_PAGES)
    visited = set()
    extracted_data = []
    
    # Process important pages first
    for url in IMPORTANT_PAGES:
        if url in visited:
            continue
        
        logger.info(f"Scraping important page: {url}")
        visited.add(url)
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch important page {url}: Status code {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            page_data = extract_text_from_page(url, soup)
            
            if page_data:
                extracted_data.append(page_data)
            
            # Find links to other pages on the same domain
            domain = urlparse(url).netloc
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Handle relative URLs
                if href.startswith('/'):
                    href = urljoin(url, href)
                # Only follow links within the same domain
                if domain in urlparse(href).netloc and href not in visited and href not in all_pages:
                    all_pages.add(href)
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
    
    # Now process other pages
    remaining_pages = all_pages - visited
    logger.info(f"Found {len(remaining_pages)} additional pages to scrape")
    
    # Limit to a reasonable number of pages
    remaining_pages = list(remaining_pages)[:100]
    
    for url in remaining_pages:
        if url in visited:
            continue
        
        logger.info(f"Scraping additional page: {url}")
        visited.add(url)
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: Status code {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            page_data = extract_text_from_page(url, soup)
            
            if page_data:
                extracted_data.append(page_data)
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
    
    logger.info(f"Scraped {len(extracted_data)} pages successfully")
    return extracted_data

# Process the extracted data into chunks for embedding
def process_data(extracted_data):
    chunks = []
    
    for page in extracted_data:
        # Process main text
        text = page["text"]
        
        # Process tables
        for table in page["tables"]:
            table_text = table_to_text(table)
            text += "\n\n" + table_text
        
        # Split text into chunks
        if len(text) > 1000:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= 1000:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk.strip(),
                            "source": page["url"],
                            "title": page["title"]
                        })
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "source": page["url"],
                    "title": page["title"]
                })
        else:
            chunks.append({
                "content": text.strip(),
                "source": page["url"],
                "title": page["title"]
            })
    
    return chunks

# Create embeddings
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings

# Main function
def main():
    logger.info("Starting web scraping process")
    
    # Scrape the website
    extracted_data = scrape_website()
    
    # Save raw data for reference
    with open(os.path.join(DATA_DIR, "na_edu_raw_data.json"), 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    # Process data into chunks
    chunks = process_data(extracted_data)
    logger.info(f"Created {len(chunks)} chunks from {len(extracted_data)} pages")
    
    # Save chunks
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Create embeddings
    logger.info("Creating embeddings")
    embeddings = create_embeddings(chunks)
    
    # Save embeddings
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    
    logger.info("Web scraping and processing completed successfully")

if __name__ == "__main__":
    main()