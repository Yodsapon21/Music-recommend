from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# โหลดข้อมูลตอนเริ่ม server
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset.csv'), index_col=0, nrows=8200)
df = df.reset_index(drop=True)

features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'speechiness']
df = df.dropna(subset=features + ['track_name', 'artists'])
df = df.reset_index(drop=True)

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])
similarity_matrix = cosine_similarity(df_scaled[features])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    song_name = request.json.get('song_name', '')
    matches = df[df['track_name'].str.lower().str.contains(song_name.lower(), na=False)]
    
    if len(matches) == 0:
        return jsonify({'error': f'ไม่พบเพลง "{song_name}"'})
    
    idx = matches.index[0]
    scores = similarity_matrix[idx]
    top = sorted([(i, float(scores[i])) for i in range(len(df)) if i != idx],
                 key=lambda x: x[1], reverse=True)[:5]
    
    results = []
    for i, score in top:
        results.append({
            'track_name': df.iloc[i]['track_name'],
            'artists': df.iloc[i]['artists'],
            'similarity': round(score, 3),
            'energy': round(float(df.iloc[i]['energy']), 2),
            'valence': round(float(df.iloc[i]['valence']), 2),
            'danceability': round(float(df.iloc[i]['danceability']), 2),
        })
    
    selected = {
        'track_name': df.iloc[idx]['track_name'],
        'artists': df.iloc[idx]['artists'],
    }
    
    return jsonify({'selected': selected, 'recommendations': results})

if __name__ == '__main__':
    app.run(debug=True)