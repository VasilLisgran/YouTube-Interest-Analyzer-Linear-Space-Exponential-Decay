import json
import math
import re
from collections import Counter
from datetime import date

PLACEHOLDER_TITLES = ('private video', 'deleted video')

STOP_WORDS = set("""
и в на с по к у о а но за из от до про для без со об при что как это все ещё уже
of the a an and to for with by at from is are was were be been being this that
these those it its they we you he she them his her our their as not no or nor but
so if then than there here where when what who why how can will would could should
just like very really quite such do does did have has had my me your
shorts short youtube video videos official live channel subscribe full hd
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep oct nov dec
monday tuesday wednesday thursday friday saturday sunday
января февраля марта апреля мая июня июля августа сентября октября ноября декабря
январь февраль март апрель май июнь июль август сентябрь октябрь ноябрь декабрь
понедельник вторник среда четверг пятница суббота воскресенье
сегодня завтра вчера год года лет день дня дней часть серия выпуск
today tomorrow yesterday year years day days part episode
one two three four five six seven eight nine ten first second third
один два три четыре пять шесть семь восемь девять десять первый второй третий
""".split())

WEIGHT_SCHEMES = {
    'original': lambda s: 10.0 if s <= 60 else 40.0 + 15.0 * math.log(s),
    'continuous_log': lambda s: 15.0 * math.log(1.0 + max(s, 1)),
    'uniform': lambda s: 1.0,
}
DEFAULT_WEIGHT_SCHEME = 'continuous_log'

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_events(path, weight_scheme=DEFAULT_WEIGHT_SCHEME, dedup_key='video_id',
                require_date=False):
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    wfun = WEIGHT_SCHEMES[weight_scheme]
    out, seen = [], set()
    for v in raw:
        title = (v.get('title') or '').strip()
        if not title or title.lower() in PLACEHOLDER_TITLES:
            continue
        key = (v.get('video_id') or title) if dedup_key == 'video_id' else title
        if key in seen:
            continue
        liked_at = v.get('liked_at')
        if require_date and not liked_at:
            continue
        seconds = v.get('duration_seconds')
        if require_date and (seconds is None or seconds < 0):
            continue
        seen.add(key)
        out.append({
            'video_id': v.get('video_id'),
            'title': title,
            'clean': clean_text(title),
            'category': v['category'],
            'date': date.fromisoformat(liked_at) if liked_at else None,
            'duration_seconds': int(seconds) if seconds is not None else None,
            'weight': float(wfun(int(seconds))) if seconds is not None else None,
        })
    if require_date:
        out.sort(key=lambda e: e['date'])
    return out

def category_document_frequency(by_category):
    df = Counter()
    for items in by_category.values():
        for w in set(t for v in items for t in v['clean'].split()):
            df[w] += 1
    return df

def keyword_score(tf_counts, df, n_categories, top_k=5):
    scored = {w: c * math.log((1 + n_categories) / (1 + df[w]))
              for w, c in tf_counts.items()}
    return [w for w, _ in sorted(scored.items(),
                                 key=lambda kv: (-kv[1], kv[0]))[:top_k]]
