"""
Awin CSV Processor
Processes Awin CSV file, generates image embeddings, and imports to Supabase.
"""

import os
import csv
import json
import hashlib
import logging
from typing import Dict, Optional, List
from urllib.parse import urlparse
import requests
from PIL import Image
import io
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('awin_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CSV_FILE = 'datafeed_2525445.csv'
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))  # Number of products to process before inserting to Supabase
MAX_PRODUCTS = int(os.getenv('MAX_PRODUCTS', '0'))  # Limit for testing (0 = no limit)
EMBEDDING_MODEL = 'google/siglip-base-patch16-384'
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize embedding model
logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Model loaded on device: {device}")


def download_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    """Download an image from URL with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"URL does not point to an image: {url}")
                return None
            
            # Load image
            image = Image.open(io.BytesIO(response.content))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to download image from {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"Failed to download image after {MAX_RETRIES} attempts: {url}")
                return None
    return None


def generate_embedding(image: Image.Image) -> Optional[List[float]]:
    """Generate embedding for an image using SigLIP model.
    
    Returns a normalized 768-dimensional embedding vector.
    """
    try:
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # Get embedding as numpy array
            embedding = outputs[0].cpu().numpy()
            
            # Normalize to unit vector (L2 normalization)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Convert to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Verify dimension (siglip-base-patch16-384 outputs 768 dimensions)
            if len(embedding_list) != 768:
                logger.warning(f"Unexpected embedding dimension: {len(embedding_list)}, expected 768")
            
            return embedding_list
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def extract_product_url(affiliate_url: str) -> str:
    """Try to extract the actual product URL from Awin affiliate link.
    
    Awin links typically have the format:
    https://www.awin1.com/pclick.php?p=PRODUCT_ID&a=AFFILIATE_ID&m=MERCHANT_ID&url=ACTUAL_URL
    
    If url parameter exists, use it. Otherwise, return the affiliate URL.
    """
    if not affiliate_url:
        return ''
    
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(affiliate_url)
        params = parse_qs(parsed.query)
        if 'url' in params:
            # Decode the actual product URL
            product_url = params['url'][0]
            # URL might be encoded, try to decode it
            try:
                from urllib.parse import unquote
                product_url = unquote(product_url)
            except:
                pass
            return product_url
    except Exception as e:
        logger.debug(f"Could not extract product URL from affiliate link: {e}")
    
    # Fallback to affiliate URL
    return affiliate_url


def normalize_product(row: Dict[str, str]) -> Dict:
    """Normalize CSV row to match Supabase table structure."""
    # Generate unique ID from source and product URL
    # Use aw_product_id as base, fallback to merchant_product_id
    product_id = row.get('aw_product_id') or row.get('merchant_product_id', '')
    
    # Get affiliate URL
    affiliate_url = row.get('aw_deep_link', '')
    source = 'awin'
    
    # Create hash-based ID if needed
    if not product_id:
        unique_string = f"{source}_{affiliate_url}"
        product_id = hashlib.md5(unique_string.encode()).hexdigest()
    
    # Try to extract actual product URL from affiliate link
    product_url = extract_product_url(affiliate_url) if affiliate_url else ''
    
    # If extraction failed, use affiliate URL as fallback
    if not product_url:
        product_url = affiliate_url
    
    # Get image URL (prefer aw_image_url, fallback to merchant_image_url)
    image_url = row.get('aw_image_url') or row.get('merchant_image_url', '')
    
    # Clean up product name (remove extra quotes)
    product_name = row.get('product_name', '').strip().strip('"')
    
    # Parse price
    price = None
    try:
        price_str = row.get('search_price') or row.get('store_price', '')
        if price_str:
            # Remove currency symbols and whitespace
            price_str = price_str.replace('USD', '').replace('$', '').strip()
            price = float(price_str)
    except (ValueError, TypeError):
        pass
    
    # Get currency
    currency = row.get('currency', 'USD')
    
    # Get brand
    brand = row.get('brand_name', '').strip()
    
    # Get category
    category = row.get('category_name') or row.get('merchant_category', '').strip()
    
    # Get size
    size = row.get('Fashion:size', '').strip()
    
    # Determine gender from product name or category (basic heuristic)
    gender = None
    product_name_lower = product_name.lower()
    if 'men' in product_name_lower or "men's" in product_name_lower:
        gender = 'male'
    elif 'women' in product_name_lower or "women's" in product_name_lower or "womens" in product_name_lower:
        gender = 'female'
    elif 'unisex' in product_name_lower:
        gender = 'unisex'
    
    # Create metadata JSON
    metadata = {
        'merchant_name': row.get('merchant_name', ''),
        'merchant_id': row.get('merchant_id', ''),
        'category_id': row.get('category_id', ''),
        'last_updated': row.get('last_updated', ''),
        'display_price': row.get('display_price', ''),
        'delivery_cost': row.get('delivery_cost', ''),
        'Fashion:category': row.get('Fashion:category', '')
    }
    
    return {
        'id': product_id,
        'source': source,
        'product_url': product_url,
        'affiliate_url': affiliate_url,
        'image_url': image_url,
        'brand': brand if brand else None,
        'title': product_name,
        'description': row.get('description', '').strip() or None,
        'category': category if category else None,
        'gender': gender,
        'price': price,
        'currency': currency if currency else None,
        'size': size if size else None,
        'second_hand': False,  # Awin products are typically new
        'country': None,  # Could be extracted from merchant or other fields if available
        'metadata': json.dumps(metadata) if any(metadata.values()) else None,
        'embedding': None  # Will be filled after image processing
    }


def process_batch(products: List[Dict]) -> List[Dict]:
    """Process a batch of products: download images and generate embeddings."""
    processed_products = []
    
    for product in tqdm(products, desc="Processing batch"):
        image_url = product.get('image_url')
        if not image_url:
            logger.warning(f"No image URL for product {product.get('id')}")
            processed_products.append(product)
            continue
        
        # Download image
        image = download_image(image_url)
        if not image:
            logger.warning(f"Could not download image for product {product.get('id')}")
            processed_products.append(product)
            continue
        
        # Generate embedding
        embedding = generate_embedding(image)
        if embedding:
            product['embedding'] = embedding
        else:
            logger.warning(f"Could not generate embedding for product {product.get('id')}")
        
        processed_products.append(product)
    
    return processed_products


def insert_to_supabase(products: List[Dict]) -> int:
    """Insert products to Supabase in batches."""
    success_count = 0
    
    # Filter out products without embeddings (or handle them separately)
    products_with_embeddings = [p for p in products if p.get('embedding')]
    products_without_embeddings = [p for p in products if not p.get('embedding')]
    
    # Insert products with embeddings
    if products_with_embeddings:
        try:
            # Supabase pgvector expects embeddings as a list/array
            # The format should be compatible with PostgreSQL vector type
            
            # Insert in smaller chunks to avoid payload size issues
            chunk_size = 20
            for i in range(0, len(products_with_embeddings), chunk_size):
                chunk = products_with_embeddings[i:i + chunk_size]
                try:
                    result = supabase.table('products').upsert(
                        chunk,
                        on_conflict='source,product_url'
                    ).execute()
                    success_count += len(chunk)
                    logger.info(f"Inserted {len(chunk)} products to Supabase")
                except Exception as e:
                    logger.error(f"Error inserting chunk to Supabase: {e}")
                    # Try inserting one by one
                    for product in chunk:
                        try:
                            supabase.table('products').upsert(
                                product,
                                on_conflict='source,product_url'
                            ).execute()
                            success_count += 1
                        except Exception as e2:
                            logger.error(f"Error inserting product {product.get('id')}: {e2}")
        except Exception as e:
            logger.error(f"Error inserting products to Supabase: {e}")
    
    # Log products without embeddings
    if products_without_embeddings:
        logger.warning(f"{len(products_without_embeddings)} products skipped due to missing embeddings")
    
    return success_count


def main():
    """Main processing function."""
    logger.info("Starting Awin CSV processing")
    if MAX_PRODUCTS > 0:
        logger.info(f"Limited to processing {MAX_PRODUCTS} products for testing")

    total_processed = 0
    total_inserted = 0
    batch = []

    # Read CSV file
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            total_rows = sum(1 for _ in reader)
            if MAX_PRODUCTS > 0:
                total_rows = min(total_rows, MAX_PRODUCTS)
            logger.info(f"Found {total_rows} rows in CSV file")

            # Reset file pointer
            f.seek(0)
            reader = csv.DictReader(f)

            for row_num, row in enumerate(tqdm(reader, total=total_rows, desc="Reading CSV")):
                # Stop if we've reached the max products limit
                if MAX_PRODUCTS > 0 and row_num >= MAX_PRODUCTS:
                    logger.info(f"Reached maximum products limit ({MAX_PRODUCTS})")
                    break

                try:
                    # Normalize product
                    product = normalize_product(row)
                    batch.append(product)

                    # Process batch when it reaches BATCH_SIZE
                    if len(batch) >= BATCH_SIZE:
                        logger.info(f"Processing batch of {len(batch)} products")
                        processed_batch = process_batch(batch)
                        inserted = insert_to_supabase(processed_batch)
                        total_inserted += inserted
                        total_processed += len(batch)
                        batch = []
                        
                        # Small delay to avoid overwhelming the API
                        time.sleep(1)
                
                except Exception as e:
                    logger.error(f"Error processing row {row_num + 1}: {e}")
                    continue
            
            # Process remaining batch
            if batch:
                logger.info(f"Processing final batch of {len(batch)} products")
                processed_batch = process_batch(batch)
                inserted = insert_to_supabase(processed_batch)
                total_inserted += inserted
                total_processed += len(batch)
    
    except FileNotFoundError:
        logger.error(f"CSV file not found: {CSV_FILE}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return
    
    logger.info(f"Processing complete!")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total inserted: {total_inserted}")


if __name__ == '__main__':
    main()

