from redis import Redis
import json

r = Redis(host='localhost', port=6379, db=0, decode_responses=True)

# string as value
r.set("user:name", "John")
retrieved_string = r.get("user:name")
print(f"Retrieved: 'user:name': '{retrieved_string}'")

# dictionary as json string
key_dict = "user-102:profile"
value_dict = {
    "name": "Bob Johnson",
    "age": 30,
    "tags": ["redis", "python", "nosql"]
}
stored_json = json.dumps(value_dict)
r.set(key_dict, stored_json)
retrieved_json = r.get(key_dict)
print(type(retrieved_json))
print(f"Retrieved Raw: '{retrieved_json}'")
retrieved_dict = json.loads(retrieved_json)
print(type(retrieved_dict))
print(f"Retrieved Dict: {retrieved_dict}")
