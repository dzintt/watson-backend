import json
import os
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from googlesearch import search

logger = logging.getLogger(__name__)

def load_cookies() -> list:
    """Load LinkedIn cookies from JSON file"""
    try:
        cookie_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "linkedin_cookies.json")
        with open(cookie_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("LinkedIn cookies file not found. Please create linkedin_cookies.json")
        return []
    except json.JSONDecodeError:
        logger.error("Invalid JSON in LinkedIn cookies file")
        return []

class LinkedIn:
    def __init__(self):
        """Initialize the LinkedIn class with Selenium WebDriver"""
        # Set up Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")  # Run in headless mode
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36")
        
        # Initialize driver as None, will be created when needed
        self.driver = None
        self.cookies = load_cookies()
        
    def _initialize_driver(self):
        """Initialize the WebDriver if not already initialized"""
        if self.driver is None:
            try:
                self.driver = webdriver.Chrome(options=self.chrome_options)
                
                # First navigate to LinkedIn to set cookies
                self.driver.get("https://www.linkedin.com")
                
                # Add cookies to the session
                for cookie in self.cookies:
                    # Some cookie attributes might cause issues, so we only set the essential ones
                    try:
                        self.driver.add_cookie({
                            'name': cookie['name'],
                            'value': cookie['value'],
                            'domain': cookie.get('domain', '.linkedin.com'),
                            'path': cookie.get('path', '/'),
                        })
                    except Exception as e:
                        logger.warning(f"Could not add cookie {cookie['name']}: {str(e)}")
                        
                # Refresh the page to apply cookies
                self.driver.refresh()
                
            except Exception as e:
                logger.error(f"Error initializing WebDriver: {str(e)}")
                raise
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {str(e)}")
            finally:
                self.driver = None
    
    def get_profile_picture(self, profile_url: str) -> dict:
        """Get the profile picture and other basic info from a LinkedIn profile"""
        try:
            self._initialize_driver()
            self.driver.get(profile_url)
            
            # Wait for the page to load
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # Extract profile information
            result = {}
            
            try:
                # Try multiple XPath patterns to find the profile picture
                selectors = [
                    "//img[contains(@class, 'pv-top-card-profile-picture__image')]",
                    "//img[contains(@class, 'profile-photo')]",
                    "//img[contains(@class, 'profile-picture')]", 
                    "//img[contains(@class, 'photo')][contains(@alt, 'profile')]",
                    "//div[contains(@class, 'profile-photo')]//img"
                ]
                
                for selector in selectors:
                    try:
                        profile_pic_elem = self.driver.find_element("xpath", selector)
                        src = profile_pic_elem.get_attribute('src')
                        if src and ('profile' in src or 'photo' in src or 'picture' in src or 'media.licdn.com' in src):
                            result['profile_picture_url'] = src
                            logger.info(f"Found profile picture using selector: {selector}")
                            break
                    except Exception:
                        continue
                        
                if 'profile_picture_url' not in result:
                    # Last resort: try to get any image that might be a profile picture
                    self.driver.save_screenshot('linkedin_debug.png')  # Save screenshot for debugging
                    logger.warning(f"Could not find profile picture with standard selectors. Saved debug screenshot.")
                    result['profile_picture_url'] = None
            except Exception as e:
                logger.warning(f"Could not extract profile picture: {str(e)}")
                result['profile_picture_url'] = None
            
            # Try to get name and headline
            try:
                # Try different selectors for name
                name_selectors = [
                    "//h1[contains(@class, 'text-heading-xlarge')]",
                    "//h1[contains(@class, 'profile-name')]",
                    "//h1[contains(@class, 'name')]"
                ]
                
                for selector in name_selectors:
                    try:
                        name_elem = self.driver.find_element("xpath", selector)
                        result['name'] = name_elem.text.strip()
                        break
                    except Exception:
                        continue
                
                # Try to get headline/title
                headline_selectors = [
                    "//div[contains(@class, 'text-body-medium')]",
                    "//div[contains(@class, 'headline')]",
                    "//div[contains(@class, 'title')]",
                    "//div[contains(@class, 'position')]"
                ]
                
                for selector in headline_selectors:
                    try:
                        headline_elem = self.driver.find_element("xpath", selector)
                        headline_text = headline_elem.text.strip()
                        if headline_text and len(headline_text) > 3:  # Avoid empty or very short headlines
                            result['headline'] = headline_text
                            break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Error extracting name/headline: {str(e)}")
            
            # Get profile URL
            result['profile_url'] = profile_url
            
            return result
            
        except Exception as e:
            logger.error(f"Error accessing LinkedIn profile {profile_url}: {str(e)}")
            return {'error': str(e), 'profile_url': profile_url}

        

def google_search_linkedin(search_query: str) -> list:
    """
    Search Google for LinkedIn profiles matching the provided query.
    
    This function leverages Google search to find LinkedIn profiles that match the given search query.
    It automatically appends 'site:linkedin.com/in' to the query to restrict results to LinkedIn profile pages.
    
    Args:
        search_query (str): The search query to find relevant LinkedIn profiles. This can be a person's name,
                           job title, company, or any other identifying information.
                           Example: "John Smith software engineer"
    
    Returns:
        list: A list of dictionaries containing search results with the following keys:
              - url (str): The URL of the LinkedIn profile
              - title (str): The title of the search result (usually contains name and headline)
              - description (str): A brief description or snippet from the profile
    """
    results = []
    for i in search(search_query + " site:linkedin.com/in", num_results=5, advanced=True):
        parsed = {
            "url": i.url,
            "title": i.title,
            "description": i.description
        }
        results.append(parsed)
    
    return results