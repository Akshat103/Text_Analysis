import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests

# Download necessary NLTK data
nltk.download('punkt')

def extract_article(url):
    """
    Extracts the title and content from a given article URL.
    """
    response = requests.get(url)
    if response.status_code != 200:
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    title = soup.find('h1', class_='entry-title')
    if not title:
        title = soup.find('h1', class_='tdb-title-text')
    
    title_text = title.get_text(strip=True) if title else None

    article_content = soup.find('div', class_='td-post-content')
    if not article_content:
        article_content = soup.find('div', class_='tdb-block-inner')
    
    article_text = article_content.get_text(strip=True) if article_content else None

    return title_text, article_text

# Read input file
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

# Directory to save extracted articles
output_dir = 'extracted_articles'
os.makedirs(output_dir, exist_ok=True)

# Extract articles and save to text files
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    print(f'Extracting article from URL ID: {url_id}')
    
    title, article_text = extract_article(url)
    if title and article_text:
        file_path = os.path.join(output_dir, f'{url_id}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(title + '\n' + article_text)
    else:
        print(f'Failed to extract article from URL ID: {url_id}')

print('Extraction complete.')

def load_stop_words(path, encoding='latin1'):
    """
    Load stop words from text files in the specified directory.
    """
    stop_words = set()
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), 'r', encoding=encoding) as f:
                words = f.read().split()
                stop_words.update(words)
    return stop_words

def load_dictionary(file_path, encoding='latin1'):
    """
    Load dictionary words from a text file.
    """
    dictionary = set()
    with open(file_path, 'r', encoding=encoding) as f:
        words = f.read().split()
        dictionary.update(words)
    return dictionary

# Load stop words and dictionaries
stop_words = load_stop_words('StopWords')
print('Completed stop word loading.')
positive_words = load_dictionary('MasterDictionary/positive-words.txt')
print('Completed positive word loading.')
negative_words = load_dictionary('MasterDictionary/negative-words.txt')
print('Completed negative word loading.')

def clean_text(text):
    """
    Cleans the input text by removing punctuation, converting to lowercase, and removing stop words.
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Analysis functions
def calculate_positive_score(tokens):
    return sum(1 for word in tokens if word in positive_words)

def calculate_negative_score(tokens):
    return sum(1 for word in tokens if word in negative_words)

def calculate_polarity_score(positive_score, negative_score):
    return (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

def calculate_subjectivity_score(positive_score, negative_score, total_words):
    return (positive_score + negative_score) / (total_words + 0.000001)

def calculate_avg_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences) if sentences else 0

def calculate_percentage_complex_words(tokens):
    complex_words = [word for word in tokens if sum(1 for char in word if char in 'aeiou') > 2]
    return len(complex_words) / len(tokens) if tokens else 0

def calculate_fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

def calculate_avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences) if sentences else 0

def count_complex_words(tokens):
    return sum(1 for word in tokens if sum(1 for char in word if char in 'aeiou') > 2)

def count_syllables(word):
    word = word.lower()
    syllables = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        syllables += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllables += 1
    if word.endswith("e"):
        syllables -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        syllables += 1
    if syllables == 0:
        syllables += 1
    return syllables

def calculate_syllables_per_word(tokens):
    return np.mean([count_syllables(word) for word in tokens])

def count_personal_pronouns(text):
    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.I)
    return len(pronouns)

def calculate_avg_word_length(tokens):
    return np.mean([len(word) for word in tokens])

def analyze_text(title, text):
    """
    Analyzes the text to calculate various linguistic features.
    """
    cleaned_tokens = clean_text(text)
    total_words = len(cleaned_tokens)
    positive_score = calculate_positive_score(cleaned_tokens)
    negative_score = calculate_negative_score(cleaned_tokens)
    polarity_score = calculate_polarity_score(positive_score, negative_score)
    subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, total_words)
    avg_sentence_length = calculate_avg_sentence_length(text)
    percentage_complex_words = calculate_percentage_complex_words(cleaned_tokens)
    fog_index = calculate_fog_index(avg_sentence_length, percentage_complex_words)
    avg_words_per_sentence = calculate_avg_words_per_sentence(text)
    complex_word_count = count_complex_words(cleaned_tokens)
    syllables_per_word = calculate_syllables_per_word(cleaned_tokens)
    personal_pronouns = count_personal_pronouns(text)
    avg_word_length = calculate_avg_word_length(cleaned_tokens)

    return {
        "POSITIVE SCORE": positive_score,
        "NEGATIVE SCORE": negative_score,
        "POLARITY SCORE": polarity_score,
        "SUBJECTIVITY SCORE": subjectivity_score,
        "AVG SENTENCE LENGTH": avg_sentence_length,
        "PERCENTAGE OF COMPLEX WORDS": percentage_complex_words,
        "FOG INDEX": fog_index,
        "AVG NUMBER OF WORDS PER SENTENCE": avg_words_per_sentence,
        "COMPLEX WORD COUNT": complex_word_count,
        "WORD COUNT": total_words,
        "SYLLABLE PER WORD": syllables_per_word,
        "PERSONAL PRONOUNS": personal_pronouns,
        "AVG WORD LENGTH": avg_word_length
    }

def extract_articles_from_dir(dir_path):
    """
    Extracts articles from text files in the specified directory.
    """
    articles = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) >= 2:
                    title = lines[0].strip()
                    text = ' '.join(lines[1:]).strip()
                    articles[filename.replace('.txt', '')] = (title, text)
    return articles

def process_and_save_results(input_excel, output_excel, articles_dir):
    """
    Processes the articles and saves the analysis results to an Excel file.
    """
    input_df = pd.read_excel(input_excel)
    articles = extract_articles_from_dir(articles_dir)
    
    results = []

    for index, row in input_df.iterrows():
        url_id = row['URL_ID']
        if url_id in articles:
            title, text = articles[url_id]
            analysis_result = analyze_text(title, text)
            analysis_result['URL_ID'] = url_id
            results.append(analysis_result)
    
    print('Completed analysis.')
    results_df = pd.DataFrame(results)
    
    output_df = input_df.merge(results_df, on='URL_ID', how='left')

    columns_order = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                    'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
                    'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                    'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
    
    output_df = output_df[columns_order]
    output_df.to_excel(output_excel, index=False)

# Define input and output file paths
input_excel = 'Input.xlsx'
output_excel = 'Output Data Structure.xlsx'
articles_dir = 'extracted_articles'

# Process articles and save results
process_and_save_results(input_excel, output_excel, articles_dir)
print(f'Results stored at {output_excel}')
