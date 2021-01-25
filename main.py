import requests
import json
from RecipePage import extract
from bs4 import BeautifulSoup
from typing import List
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  

URL = 'https://www.cocinafacil.com.mx/recetas/recetas-para-hot-cakes/'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')

results = soup.find(id='post-27696')

liElements = results.find_all('li', class_='simmer-connected-recipe')

anchors = [x.find_all('a') for x in liElements]
hrefs = [x[0].get('href') for x in anchors]

hrefList: List[str] = hrefs

print('Recetas: ')
print(hrefList)

recipePages = map(extract, hrefList)
res = list(recipePages)
corpus = [" ".join(x['ingredients']) + " ".join(x['instructions']) for x in res]
namesFromCorpus = [x.split(" ") for x in corpus]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
names = vectorizer.get_feature_names()
farray = X.toarray().tolist()
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(farray)
tfidfArray = tfidf.toarray().tolist() 

collection = []
for i in range(len(tfidfArray)):
    collection.append({"recipe": res[i]['title'], "words":[] })
    for y in range(len(tfidfArray[i])):
        collection[i]['words'].append({ "name": names[y], "tf-idf": tfidfArray[i][y] })
    collection[i]['words'].sort(reverse=True,key=lambda x: x['tf-idf'])

tfLazyResults = [{**z, "words":filter(lambda y: y['tf-idf'] != 0, z['words'])} for z in collection]
tfResults = [{**x, "words":list(x['words'])} for x in tfLazyResults]

wordTotal = {}
for i in range(len(tfResults)):
    for j in range(len(tfResults[i]['words'])):
        currentKey = tfResults[i]['words'][j]['name']
        if(not currentKey in wordTotal):
            wordTotal[currentKey] = tfResults[i]['words'][j]['tf-idf']
        else:
            wordTotal[currentKey] = tfResults[i]['words'][j]['tf-idf'] + wordTotal[currentKey]

sortedWordTotal = dict(sorted(wordTotal.items(), key=lambda item: item[1], reverse=True))

pagesData = {
    "data": tfResults
}

with open("PagesData.json","w") as outfile:
    json.dump(tfResults,outfile)

with open("Data.json", "w") as outfile:
    json.dump(wordTotal, outfile)

print(sortedWordTotal)
