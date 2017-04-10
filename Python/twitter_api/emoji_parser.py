import json, re
content = []

with open('amazon_dataset.json','r') as file:
    lines = [line.rstrip('\n') for line in file]

# dict_input = json.load('amazon_dataset.json')
# print('!!!!!Type of DATA: '+str(type(content)))
lst = []
for i in range(len(lines)):
    if i%2 == 0:
        lst.append(json.loads(lines[i]))

emoji_pattern = re.compile('[\U0001F300-\U0001F64F]')

for dict_input in lst:
    # dict_input = json.loads(json_input)
    text = dict_input['text']
    screen_name = dict_input['user']['screen_name']
    emojis = emoji_pattern.findall(text)

    print(len(emojis), 'chars found in post by', screen_name)
    for emoji in emojis:
        print('emoji: ' + json.dumps(emoji))