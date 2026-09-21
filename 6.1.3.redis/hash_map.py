import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 1. Create a Hash (HSET)
# Store multiple fields for a single "User" object
r.hset('user:1001', mapping={
    'name': 'John Doe',
    'email': 'john@example.com',
    'login_count': 5,
    'status': 'active'
})

# 2. Get a single field (HGET)
user_name = r.hget('user:1001', 'name')
print(f"User Name: {user_name}")

# 3. Get all fields and values (HGETALL)
# This returns a Python dictionary
user_data = r.hgetall('user:1001')
print(f"All User Data: {user_data}")

# 4. Increment a numeric field (HINCRBY)
# Perfect for counters like page views or login attempts
r.hincrby('user:1001', 'login_count', 1)
new_count = r.hget('user:1001', 'login_count')
print(f"Updated Login Count: {new_count}")

# 5. Check if a field exists (HEXISTS)
has_email = r.hexists('user:1001', 'email')
print(f"Has email field? {has_email}")
