import requests
import time
import os
import sys
import urllib3
import concurrent.futures

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TARGET_SIZE_MB = 10
OUTPUT_FILE = "data/corpus.txt"
API_URL = "https://simple.wikipedia.org/w/api.php"
NUM_WORKERS = 20

def get_random_articles_batch():
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": 20,
        "prop": "extracts",
        "explaintext": 1,
        "exintro": 1 
    }
    del params["exintro"]

    try:
        headers = {
            'User-Agent': 'SimpleLLMScraper/1.0 (rishu@example.com)'
        }
        response = requests.get(API_URL, params=params, headers=headers, verify=False, timeout=30)
        data = response.json()
        
        articles = []
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            title = page_data.get("title", "")
            extract = page_data.get("extract", "")
            
            # Cleaning and Filtering
            if not extract or len(extract) < 1000: # Filter short stubs
                continue
                
            # Filter out noise sections (rudimentary)
            lines = extract.split('\n')
            clean_lines = []
            for line in lines:
                if line.strip().startswith("==") and any(x in line for x in ["References", "Other websites", "Related pages", "See also"]):
                    continue
                clean_lines.append(line)
            
            clean_extract = "\n".join(clean_lines).strip()
            
            if len(clean_extract) > 500:
                articles.append(f"\n\n== {title} ==\n\n{clean_extract}")
                
        return articles
            
    except Exception as e:
        # print(f"Error fetching batch: {e}")
        return []

def fetch_worker(_):
    return get_random_articles_batch()

def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    current_size = 0
    target_bytes = TARGET_SIZE_MB * 1024 * 1024
    
    print(f"Scraping Simple Wikipedia (Parallel) until {TARGET_SIZE_MB}MB...")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while current_size < target_bytes:
                # Submit jobs
                futures = [executor.submit(fetch_worker, i) for i in range(NUM_WORKERS)]
                
                for future in concurrent.futures.as_completed(futures):
                    batch = future.result()
                    if not batch:
                        continue
                    
                    for article in batch:
                        f.write(article)
                    
                    f.flush()
                    current_size = os.path.getsize(OUTPUT_FILE)
                    mb_size = current_size / (1024 * 1024)
                    sys.stdout.write(f"\rProgress: {mb_size:.2f}/{TARGET_SIZE_MB} MB")
                    sys.stdout.flush()
                    
                    if current_size >= target_bytes:
                        executor.shutdown(wait=False)
                        break

    print(f"\nDone! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
