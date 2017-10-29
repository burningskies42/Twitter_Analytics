from os import getcwd

pth = getcwd()+'\\amazon_labeled.json'

with open(pth,'r') as fid:
   a = fid.read()

a = a.split('\n\n')
#
# for i in a:
#    print(i)

pth = getcwd()+'\\amazon_labeled_list.json'
with open(pth,'w') as fid:
   for line in a:
      fid.write(line+',')