import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

def extract_article(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    title = soup.find('h1', class_='entry-title')
    if title is None:
        title = soup.find('h1', class_='tdb-title-text')
    
    title_text = title.get_text(strip=True) if title else None

    article_content = soup.find('div', class_='td-post-content')
    if article_content is None:
        article_content = soup.find('div', class_='tdb-block-inner')
    
    article_text = article_content.get_text(strip=True) if article_content else None

    return title_text, article_text

input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

output_dir = 'extracted_articles'
os.makedirs(output_dir, exist_ok=True)

for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    print(f'Extracting article from URL ID: {url_id}, URL: {url}')
    
    title, article_text = extract_article(url)
    if title and article_text:
        file_path = os.path.join(output_dir, f'{url_id}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(title + '\n' + article_text)
    else:
        print(f'Failed to extract article from URL ID: {url_id}')

print('Extraction complete.')
