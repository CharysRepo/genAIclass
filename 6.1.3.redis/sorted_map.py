import redis

# 1. Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# 2. Add elements with scores (ZADD) - e.g., Player Scores
# Members are unique, scores can be updated
r.zadd('leaderboard', {'Alice': 100, 'Bob': 400, 'Charlie': 300, 'David': 200})

# 3. Retrieve ranked members (ascending)
print("Lowest score to highest:", r.zrange('leaderboard', 0, -1, withscores=True))

# 4. Retrieve ranked members (descending)
print("Highest score to lowest:", r.zrevrange('leaderboard', 0, -1, withscores=True))

# 5. Update score (ZADD again)
r.zadd('leaderboard', {'Alice': 500})
print("Updated order:", r.zrevrange('leaderboard', 0, -1, withscores=True))

# 6. Get rank/position of a member (0-indexed)
david_rank = r.zrank('leaderboard', 'David')
david_rev_rank = r.zrevrank('leaderboard', 'David')
david_score = r.zscore('leaderboard', 'David')
print(f"David's rank: {david_rank}")
print(f"David's reverse rank: {david_rev_rank}")
print(f"David's score: {david_score}")

