import requests
import nltk
import re
from bs4 import BeautifulSoup
from typing import List
from nltk.corpus import stopwords
from unicodedata import normalize

def cleanNFD(s):
    return re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
        normalize("NFD", s), 0, re.I
    )


def extract(URL):
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')

    results = soup.find(id='main')

    ingredientElements = results.find_all('li', class_='simmer-ingredient')
    instructionElements = results.find_all('li', class_='simmer-instruction')

    def extractIngredients(ingredients):
        spans = [x.find_all('span') for x in ingredients]
        spanValues = [[cleanNFD(y.string) for y in x] for x in spans]
        tokens = [[nltk.word_tokenize(y) for y in x] for x in spanValues]
        filtered_tokens = [[[w for w in x if not w.lower() in stopwords.words('spanish')] for x in y] for y in tokens]
        joinedValues = [[" ".join(y) for y in x] for x in filtered_tokens]
        joinedSequences = [" ".join(x) for x in joinedValues]
        return joinedSequences

    ingredientList = extractIngredients(ingredientElements)
    cleanInstructions = [cleanNFD(x.string) for x in instructionElements]
    instructionTokens = [nltk.word_tokenize(x) for x in cleanInstructions]
    sanitizedInstructions = [[w for w in x if not w.lower() in stopwords.words('spanish')] for x in instructionTokens] 
    joinedInstructions = [" ".join(x) for x in sanitizedInstructions]

    result = {
        "title": cleanNFD(soup.title.string),
        "ingredients": ingredientList,
        "instructions": joinedInstructions
    }

    return result