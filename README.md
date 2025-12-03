# Awin CSV to Supabase Processor

This script processes Awin CSV data feed files, generates image embeddings using Google's SigLIP model, and imports the data to Supabase.

## Features

- Reads Awin CSV data feed
- Downloads product images from URLs
- Generates image embeddings using `google/siglip-base-patch16-384` model
- Normalizes and maps CSV columns to Supabase table structure
- Batch processing for efficient import
- Error handling and retry logic
- Progress tracking with tqdm

## Prerequisites

- Python 3.8 or higher
- Supabase account and database
- CUDA-capable GPU (optional, but recommended for faster embedding generation)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file in the project root with:
```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

## Usage

1. Place your Awin CSV file (`datafeed_2525445.csv`) in the project root directory.

2. Run the script:
```bash
python process_awin_csv.py
```

The script will:
- Read all rows from the CSV
- Process products in batches (default: 50 products per batch)
- Download images and generate embeddings
- Insert data into your Supabase `products` table

## Configuration

You can modify these constants in `process_awin_csv.py`:
- `BATCH_SIZE`: Number of products to process before inserting to Supabase (default: 50)
- `MAX_RETRIES`: Number of retry attempts for image downloads (default: 3)
- `RETRY_DELAY`: Delay between retries in seconds (default: 2)

## CSV Column Mapping

The script maps Awin CSV columns to Supabase table columns as follows:

| Awin CSV Column | Supabase Column | Notes |
|----------------|-----------------|-------|
| `aw_product_id` | `id` | Primary key |
| `aw_deep_link` | `affiliate_url` | Affiliate tracking URL |
| `aw_deep_link` | `product_url` | Used as product URL |
| `aw_image_url` or `merchant_image_url` | `image_url` | Prefers aw_image_url |
| `product_name` | `title` | Cleaned of extra quotes |
| `description` | `description` | |
| `brand_name` | `brand` | |
| `category_name` or `merchant_category` | `category` | |
| `Fashion:size` | `size` | |
| `search_price` or `store_price` | `price` | Parsed as float |
| `currency` | `currency` | |
| Auto-detected from title | `gender` | Heuristic-based |
| Various | `metadata` | JSON object with additional fields |

## Database Schema

Ensure your Supabase table matches this structure:

```sql
create table public.products (
  id text not null,
  source text null,
  product_url text null,
  affiliate_url text null,
  image_url text not null,
  brand text null,
  title text not null,
  description text null,
  category text null,
  gender text null,
  price double precision null,
  currency text null,
  search_tsv tsvector null,
  created_at timestamp with time zone null default now(),
  metadata text null,
  size text null,
  second_hand boolean null default false,
  embedding public.vector null,
  country text null,
  constraint products_pkey primary key (id),
  constraint products_source_product_url_key unique (source, product_url)
);
```

**Important**: Make sure the `embedding` column is of type `vector` (pgvector extension) with the appropriate dimensions. For `google/siglip-base-patch16-384`, the embedding dimension is **768**.

To set up the vector column:
```sql
-- Install pgvector extension if not already installed
CREATE EXTENSION IF NOT EXISTS vector;

-- Update the embedding column to use vector type with correct dimensions (768 for siglip-base-patch16-384)
ALTER TABLE products 
ALTER COLUMN embedding TYPE vector(768);

-- Create an index on the embedding column for efficient similarity search (optional but recommended)
CREATE INDEX ON products USING ivfflat (embedding vector_cosine_ops);
```

**Note**: The script normalizes all embeddings to unit vectors (L2 normalization) for better similarity search performance.

## Logging

The script creates a log file `awin_processor.log` with detailed information about the processing, including:
- Progress updates
- Errors and warnings
- Success/failure counts

## Performance Notes

- Processing ~47,000 products will take several hours depending on:
  - Network speed (for image downloads)
  - GPU availability (for embedding generation)
  - Supabase connection speed
- The script processes images sequentially to avoid overwhelming servers
- Consider running during off-peak hours for large datasets

## Troubleshooting

1. **Out of memory errors**: Reduce `BATCH_SIZE` or process the CSV in chunks
2. **Image download failures**: Check network connectivity and image URL accessibility
3. **Supabase connection errors**: Verify your `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
4. **Embedding dimension mismatch**: Ensure your Supabase `embedding` column is set to `vector(768)`

## License

MIT

