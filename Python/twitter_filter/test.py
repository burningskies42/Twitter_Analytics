
ignore_terms = input("ignore tweets containing (optional, semicolon-separated): ")
ignore_terms = [w.replace(' ','') for w in ignore_terms.split(';') if len(w.replace(' ',''))>0]
print(ignore_terms)

if any(ignore_str in text for ignore_str in ignore_terms):
    print('true')
    quit()

print('none found')
