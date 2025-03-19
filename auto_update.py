import schedule
import time
import os
import sys
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("updater.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

def run_scraper():
    """Run the enhanced scraper script to update the knowledge base"""
    logger.info(f"Starting scheduled update at {datetime.now()}")
    
    try:
        # Run the enhanced scraper
        scraper_path = os.path.join(script_dir, "enhanced_scraper.py")
        result = subprocess.run([sys.executable, scraper_path], 
                                capture_output=True, 
                                text=True)
        
        if result.returncode == 0:
            logger.info("Scraper ran successfully")
            logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"Scraper failed with return code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
    
    except Exception as e:
        logger.error(f"Error running scraper: {str(e)}")
    
    logger.info(f"Completed scheduled update at {datetime.now()}")

def main():
    logger.info("Starting auto-update scheduler")
    
    # Schedule the scraper to run once a week (or adjust as needed)
    schedule.every().monday.at("03:00").do(run_scraper)
    
    # Also run immediately on startup
    run_scraper()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()