import csv
from pathlib import Path
from urllib.parse import urlparse
import asyncpg
from config.config_reader import settings

class IngestionService:
    def __init__(self):
        """Initializes the service with an active connection pool."""
        self.pool = None
        self.dsn = f"postgresql://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DATABASE}"

    async def __aenter__(self) -> "IngestionService":
        """Ensures the database exists, initializes the pool, and sets up schema."""
        await self._create_database_if_not_exists(self.dsn)
        self.pool = await asyncpg.create_pool(self.dsn)
        await self.setup_schema()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exits the runtime context and safely closes the connection pool."""
        if self.pool:
            await self.pool.close()
            print("🛑 Connection pool closed successfully.")

    @staticmethod
    async def _create_database_if_not_exists(dsn:str):
        """Extracts DB name from settings and creates it if it doesn't exist."""
        url = urlparse(dsn)
        db_name = url.path.lstrip('/')
        if not db_name: 
            return

        admin_url = url._replace(path="/postgres").geturl()
        try:
            conn = await asyncpg.connect(admin_url)
            exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
            if not exists:
                await conn.execute(f'CREATE DATABASE {db_name}')
                print(f"✨ Database '{db_name}' created.")
            await conn.close()            
        except asyncpg.exceptions.InsufficientPrivilegeError:
            print(f"❌ Error: User lacks CREATEDB permissions for '{db_name}'.")
        except Exception as e:
            print(f"❌ Database creation failed: {e}")

    async def setup_schema(self):
        """Creates all tables in the database using the internal pool."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                queries = [
                    "CREATE TABLE IF NOT EXISTS customers (customer_id TEXT PRIMARY KEY, customer_unique_id TEXT, customer_zip_code_prefix INTEGER, customer_city TEXT, customer_state TEXT)",
                    "CREATE TABLE IF NOT EXISTS sellers (seller_id TEXT PRIMARY KEY, seller_zip_code_prefix INTEGER, seller_city TEXT, seller_state TEXT)",
                    "CREATE TABLE IF NOT EXISTS products (product_id TEXT PRIMARY KEY, product_category_name TEXT, product_name_lenght REAL, product_description_lenght REAL, product_photos_qty REAL, product_weight_g REAL, product_length_cm REAL, product_height_cm REAL, product_width_cm REAL)",
                    "CREATE TABLE IF NOT EXISTS geolocation (geolocation_zip_code_prefix INTEGER, geolocation_lat REAL, geolocation_lng REAL, geolocation_city TEXT, geolocation_state TEXT)",
                    "CREATE TABLE IF NOT EXISTS product_category_name_translation (product_category_name TEXT, product_category_name_english TEXT)",
                    "CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, customer_id TEXT REFERENCES customers(customer_id), order_status TEXT, order_purchase_timestamp TEXT, order_approved_at TEXT, order_delivered_carrier_date TEXT, order_delivered_customer_date TEXT, order_estimated_delivery_date TEXT)",
                    "CREATE TABLE IF NOT EXISTS order_items (order_id TEXT REFERENCES orders(order_id), order_item_id INTEGER, product_id TEXT REFERENCES products(product_id), seller_id TEXT REFERENCES sellers(seller_id), shipping_limit_date TEXT, price REAL, freight_value REAL, PRIMARY KEY (order_id, order_item_id))",
                    "CREATE TABLE IF NOT EXISTS order_payments (order_id TEXT REFERENCES orders(order_id), payment_sequential INTEGER, payment_type TEXT, payment_installments INTEGER, payment_value REAL)",
                    "CREATE TABLE IF NOT EXISTS order_reviews (review_id TEXT, order_id TEXT REFERENCES orders(order_id), review_score INTEGER, review_comment_title TEXT, review_comment_message TEXT, review_creation_date TEXT, review_answer_timestamp TEXT)"
                ]
                for query in queries:
                    await conn.execute(query)
        print("✅ Full schema initialized.")

    async def import_csv(self, table_name: str, file_path: str | Path):
        """Imports CSV using the high-speed COPY protocol via the internal pool."""
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ Skipping {table_name}: {file_path} not found.")
            return

        int_cols = {'customer_zip_code_prefix', 'seller_zip_code_prefix', 'order_item_id', 
                    'payment_sequential', 'payment_installments', 'review_score', 'geolocation_zip_code_prefix'}
        float_cols = {'price', 'freight_value', 'payment_value', 'product_name_lenght', 
                      'product_description_lenght', 'product_photos_qty', 'product_weight_g', 
                      'product_length_cm', 'product_height_cm', 'product_width_cm', 
                      'geolocation_lat', 'geolocation_lng'}

        async with self.pool.acquire() as conn:
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                try:
                    headers = next(reader)
                except StopIteration: 
                    return

                data = []
                for row in reader:
                    cleaned = []
                    for i, cell in enumerate(row):
                        val = cell.strip()
                        if val == "" or val.lower() == "nan": 
                            cleaned.append(None)
                        elif headers[i] in int_cols: 
                            cleaned.append(int(float(val)))
                        elif headers[i] in float_cols: 
                            cleaned.append(float(val))
                        else: 
                            cleaned.append(val)
                    data.append(tuple(cleaned))

                try:
                    await conn.copy_records_to_table(
                        table_name, 
                        records=data, 
                        columns=headers, 
                        timeout=60.0
                    )
                    print(f"🚀 High-speed COPY complete: {len(data)} rows -> '{table_name}'.")
                except asyncpg.UniqueViolationError:
                    print(f"❌ COPY failed for '{table_name}': Duplicate keys found. (COPY does not support ON CONFLICT)")
                except Exception as e:
                    print(f"❌ COPY failed for '{table_name}': {e}")

    async def run_pipeline(self):
        """Executes the entire workflow mapping for files in the configured directory."""
        data_folder = Path(settings.DATA_DIR)
        csv_mappings = [
            ("customers", data_folder / "customers.csv"),
            ("sellers", data_folder / "sellers.csv"),
            ("products", data_folder / "products.csv"),
            ("geolocation", data_folder / "geolocation.csv"),
            ("orders", data_folder / "orders.csv"),
            ("order_items", data_folder / "order_items.csv"),
            ("order_payments", data_folder / "order_payments.csv"),
            ("order_reviews", data_folder / "order_reviews.csv"),
            ("product_category_name_translation", data_folder / "product_category_name_translation.csv"),
        ]

        for table, file in csv_mappings:
            await self.import_csv(table, file)


