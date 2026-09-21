import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 1. Add items to the list (LPUSH / RPUSH)
# Imagine a simple message queue or notification feed
r.rpush('notifications:user:1001', 'Welcome to the app!')
r.rpush('notifications:user:1001', 'Your profile is 50% complete.')
r.lpush('notifications:user:1001', 'URGENT: Login detected from new device!') # Adds to the front

# 2. Get a range of items (LRANGE)
# 0 is the first item, -1 is the last item
all_notifs = r.lrange('notifications:user:1001', 0, -1)
print(f"User Notifications: {all_notifs}")

# 3. Get the length of the list (LLEN)
count = r.llen('notifications:user:1001')
print(f"Total notifications: {count}")

# 4. Remove and return items (LPOP / RPOP)
# This acts like a queue (First-In, First-Out)
first_in = r.lpop('notifications:user:1001')
print(f"Processing notification: {first_in}")

# 5. Trim the list (LTRIM)
# Keep only the most recent 2 items (useful for fixed-size logs)
r.ltrim('notifications:user:1001', 0, 1)
remaining = r.lrange('notifications:user:1001', 0, -1)
print(f"Remaining (trimmed): {remaining}")
