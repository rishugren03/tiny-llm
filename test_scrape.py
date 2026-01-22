import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
url = "https://simple.wikipedia.org/wiki/Special:Random"

try:
    print(f"Fetching {url}...")
    response = requests.get(url, verify=False, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Content length: {len(response.content)}")
    print(f"Preview: {response.content[:100]}")
except Exception as e:
    print(f"Error: {e}")
