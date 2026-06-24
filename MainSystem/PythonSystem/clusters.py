from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json
import re

model = SentenceTransformer('sentence-transformers/LaBSE')


def clean_text(text):
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    
    text = re.sub(r'\b\d+\b', ' ', text)  # numbers
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# reading from the folder data
with open('../data/user_videos.json', 'r', encoding='utf-8') as f:
    videos = json.load(f)

# Groupping by categories
videos_by_category = defaultdict(list)
for video in videos:
    videos_by_category[video["category"]].append(video["title"])


stop_words = {
    '-', 'и', 'в', 'на', 'с', 'по', 'к', 'у', 'о', 'а', 'но', 'за', 'из', 'от', 'до',
    'of', 'the', 'a', 'an', 'and', 'to', 'for', 'with', 'by', 'at', 'from', 'is', 'was',
    'no', 'woman', 'cry', 'birds', 'this', 'that', 'these', 'those', 'it', 'they', 'we',
    'you', 'he', 'she', 'them', 'can', 'will', 'would', 'could', 'should', 'just', 'like',
    'very', 'really', 'quite', 'such', 'there', 'here', 'where', 'when', 'what', 'who',
    'why', 'how', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'so',
    'or', 'nor', 'yet', 'because', 'shorts', 'youtube', 'video', 'official', 'live',
    'new', 'best', 'top', '2024', '2025', '2026', '2023', '1', '2', '3', '4', '5', '', 'nooo', 'врек', 
    'youtubeshorts', '', 'real', ' ',

    'января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 
    'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря',
    'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
    'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь',
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    
    'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    
    'зима', 'весна', 'лето', 'осень', 'winter', 'spring', 'summer', 'autumn', 'fall',
    
    'сегодня', 'завтра', 'вчера', 'год', 'года', 'лет', 'день', 'дня', 'дней',
    'today', 'tomorrow', 'yesterday', 'year', 'years', 'day', 'days',
    
    'марта', 'декабря', '2026', '2025', '2024', '2023', '2022',

    '1x6', 'дахака', 'раз', 'два', 'три', 
    'минута', 'секунда', 'час', 'минут', 'секунд', 
    
    'walkthrough', 'playthrough', 'episode', 'part', 
    'часть', 'серия', 'выпуск',
    
    'про', 'для', 'без', 'до', 'по', 'со', 'из', 'у', 'о', 'об', 'при',

    'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять',
    'одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать', 'пятнадцать',
    'двадцать', 'тридцать', 'сорок', 'пятьдесят', 'сто', 'тысяча', 'тысяч',
    'первый', 'второй', 'третий', 'четвертый', 'пятый', 'шестой', 'седьмой', 'восьмой', 'девятый', 'десятый',
    'первая', 'вторая', 'третья', 'четвертая', 'пятая',
    'одно', 'две', 'трое',
    
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
    'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
}

# Create clusters
final = {}

for category, titles in videos_by_category.items():

    # We need more than 2 videos to make a cluster
    if len(titles) < 2:
        print("< 2 videos")
        continue
    
    embeddings = model.encode(titles)

    clustering = DBSCAN(eps=0.51, min_samples=2, metric='cosine')
    labels = clustering.fit_predict(embeddings)

    # Groupping clusters by labels
    clusters = defaultdict(list)
    for title, label in zip(titles, labels):
        clusters[label].append(title)

    # Printing labels and clusters
    for label, titles in sorted(clusters.items()):
        print(label, " ", titles)

    category_clusters = {}

    # Titles -> Words 
    for label, titles in clusters.items():
        if label == -1:
            continue

        words_count = defaultdict(int)
        
        for title in titles:
            cleanTitle = clean_text(title)
            words = cleanTitle.split()

            for word in words:
                if word not in stop_words and len(word) > 2:
                    words_count[word]+=1

        top_words = sorted(words_count.items(), key=lambda x: x[1], reverse=True)[:5]
        top_words_list = [word for word, count in top_words]
        if category in ["Gaming", "News & Politics"]:
            top_words_list = top_words_list[:2]
        print(top_words)

        if len(top_words_list)>0:  # only if not empty
            category_clusters[str(label)] = top_words_list
        
    final[category] = category_clusters

    print('\n ######################################## \n')

# Save back to the data folder
with open('../data/clusters_result.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print("✅ Ready! ../data/clusters_result.json")
