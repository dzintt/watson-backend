import requests
import json
from googlesearch import search

def load_cookies() -> dict:
    with open("linkedin_cookies.json", "r") as f:
        return json.load(f)

class LinkedIn:
    def __init__(self):
        self.cookies = load_cookies()
        self.headers = {
            "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
        }
        self.session = requests.Session()

        for cookie in self.cookies:
            self.session.cookies.set(cookie["name"], cookie["value"])

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
    
    Note:
        - Results are limited to the first page of Google search results
        - This method depends on the googlesearch-python package
        - No authentication with LinkedIn is required for this method
        - The quality of results depends on Google's search algorithm and indexing
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