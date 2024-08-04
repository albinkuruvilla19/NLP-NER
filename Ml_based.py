#755a8c08f4dc4ca9bf064f122c0241de
import requests
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# NewsAPI details
API_KEY = '755a8c08f4dc4ca9bf064f122c0241de'
NEWS_API_URL = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}'

def fetch_news_article():
    response = requests.get(NEWS_API_URL)
    articles = response.json().get('articles')
    if not articles:
        raise ValueError("No articles found")
    return articles[0]['content']

def extract_entities_spacy(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_entities_nltk(text):
    sentences = nltk.sent_tokenize(text)
    entities = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        for chunk in chunks:
            if isinstance(chunk, Tree):
                entity = " ".join(c[0] for c in chunk)
                entity_type = chunk.label()
                entities.append((entity, entity_type))
    return entities

def main():
    article_text = fetch_news_article()
    print("Article Text: ", article_text)

    spacy_entities = extract_entities_spacy(article_text)
    print("\nEntities extracted by spaCy:")
    for entity in spacy_entities:
        print(entity)

    nltk_entities = extract_entities_nltk(article_text)
    print("\nEntities extracted by NLTK:")
    for entity in nltk_entities:
        print(entity)

    # Compare results
    spacy_entity_set = set(spacy_entities)
    nltk_entity_set = set(nltk_entities)
    common_entities = spacy_entity_set & nltk_entity_set
    unique_spacy_entities = spacy_entity_set - nltk_entity_set
    unique_nltk_entities = nltk_entity_set - spacy_entity_set

    print("\nCommon Entities:")
    for entity in common_entities:
        print(entity)

    print("\nUnique Entities in spaCy:")
    for entity in unique_spacy_entities:
        print(entity)

    print("\nUnique Entities in NLTK:")
    for entity in unique_nltk_entities:
        print(entity)

if __name__ == "_main_":
    main()