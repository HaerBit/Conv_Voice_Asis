import json

with open('Dict_ChangeText.json','r',encoding='utf-8') as file:
    dict = json.load(file)
def Replacing_The_Text(text):
    text_split = text.split() +['']
    for index in range(1,len(text_split)):
        comp_word = " ".join(text_split[index-1:index+1])
        if text_split[index-1] in dict:
            text_split[index-1] = dict[text_split[index-1]]
        elif comp_word in dict:
            text_split[index-1]=dict[comp_word]
            text_split[index] = ''
    return text_split
