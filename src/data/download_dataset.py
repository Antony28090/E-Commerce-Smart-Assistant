import pandas as pd
from datasets import load_dataset
import os

def download_and_save_dataset():
    print("Loading Amazon Reviews dataset from Stanford SNAP...")
    url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
    
    # Read the first 10,000 lines from the gzipped JSON
    df = pd.read_json(url, lines=True, nrows=10000)
    
    # The columns are: reviewerID, asin, reviewerName, helpful, reviewText, overall, summary, unixReviewTime, reviewTime
    # We map 'overall' to 'rating', 'summary' to 'review_title', 'reviewText' to 'text'
    df.rename(columns={
        'overall': 'rating',
        'summary': 'review_title',
        'reviewText': 'text'
    }, inplace=True)
    
    df['category'] = 'Electronics'
    
    # Keep only relevant columns
    columns_to_keep = ['rating', 'category', 'review_title', 'text']
    df = df[columns_to_keep]
    
    # Create the target directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'reviews.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Dataset successfully downloaded and saved to {output_file}")
    print(f"Total reviews: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    download_and_save_dataset()
