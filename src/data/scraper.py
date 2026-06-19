import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

BASE_URL = 'http://books.toscrape.com/catalogue/category/books_1/'

def scrape_books(max_pages=5):
    print("Starting data mining pipeline using BeautifulSoup...")
    books_data = []
    
    for page in range(1, max_pages + 1):
        url = f'{BASE_URL}page-{page}.html'
        print(f"Scraping page {page}: {url}")
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all books on the page
        books = soup.find_all('article', class_='product_pod')
        
        for book in books:
            # Extract title
            title = book.h3.a['title']
            
            # Extract rating (class: star-rating One/Two/Three/Four/Five)
            rating_class = book.find('p', class_='star-rating')['class'][1]
            rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
            rating = rating_map.get(rating_class, 0)
            
            # Since the main page doesn't have the full description/review,
            # we would ideally fetch the book detail page. For simplicity/rate limits:
            book_url = 'http://books.toscrape.com/catalogue/' + book.h3.a['href'].replace('../', '')
            
            # Simulated basic rate limiting
            time.sleep(0.5) 
            
            try:
                detail_response = requests.get(book_url)
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                
                # Extract category
                category = detail_soup.find('ul', class_='breadcrumb').find_all('li')[2].text.strip()
                
                # Extract product description (simulating review text)
                product_desc_tag = detail_soup.find('div', id='product_description')
                description = product_desc_tag.find_next_sibling('p').text if product_desc_tag else ""
                
                books_data.append({
                    'review_title': title,
                    'rating': rating,
                    'category': category,
                    'text': description,
                    'timestamp': pd.Timestamp.now() # Simulated timestamp as the site doesn't have reviews
                })
            except Exception as e:
                print(f"Error scraping details for {title}: {e}")
                
    print(f"Scraped {len(books_data)} items successfully.")
    
    # Structure raw output into a clean CSV
    df = pd.DataFrame(books_data)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'scraped_reviews.csv')
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    # In a real scenario, max_pages would be higher to get 5,000+ items.
    # books.toscrape.com only has 1000 books total (50 pages).
    scrape_books(max_pages=2)
