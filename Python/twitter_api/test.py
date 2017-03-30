import pandas as pd
import json

json_data=open('amazon_db.json').read()

data = json.loads(json_data)
print(data)