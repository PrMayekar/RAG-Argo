import json
import os
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# ── Attempt real DB/RAG imports; fall back to demo data ──────────────────────
try:
    from sqlalchemy import create_engine
    import pandas as pd
    from dotenv import load_dotenv
    load_dotenv()
    DB_URL = (
        f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}"
        f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME','argo_db')}"
    )
    engine = create_engine(DB_URL)
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils import embedding_functions
    from groq import Groq
    CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
    EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# ── ML imports ───────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── DEMO DATA (used when DB is not connected) ────────────────────────────────
def generate_demo_data():
    np.random.seed(42)
    n = 120
    cycles = np.repeat(np.arange(1, 13), 10)
    base_lat = -40 + np.cumsum(np.random.randn(12) * 0.3)
    base_lon = 60 + np.cumsum(np.random.randn(12) * 0.5)
    lats = np.repeat(base_lat, 10) + np.random.randn(n) * 0.05
    lons = np.repeat(base_lon, 10) + np.random.randn(n) * 0.05
    depth = np.tile(np.linspace(0, 2000, 10), 12)
    pressure = depth * 1.025
    temp = 25 - depth / 100 + np.random.randn(n) * 0.3
    sal = 35 + np.random.randn(n) * 0.1

    import pandas as pd
    return pd.DataFrame({
        'cycle_number': cycles,
        'latitude': lats,
        'longitude': lons,
        'pressure': pressure,
        'temperature': temp,
        'salinity': sal,
        'profile_date': pd.date_range('2024-01-01', periods=12, freq='10D').repeat(10),
    })

def get_df(float_wmo=None):
    if DB_AVAILABLE:
        try:
            import pandas as pd
            if float_wmo:
                query = f"""
                    SELECT m.*, p.latitude, p.longitude, p.cycle_number, p.profile_date
                    FROM measurements m
                    JOIN profiles p ON m.profile_id = p.id
                    JOIN argo_floats f ON f.id = p.float_id
                    WHERE f.wmo_id = '{float_wmo}'
                    ORDER BY p.cycle_number, m.depth_level
                """
            else:
                query = """
                    SELECT m.*, p.latitude, p.longitude, p.cycle_number, p.profile_date
                    FROM measurements m
                    JOIN profiles p ON m.profile_id = p.id
                    ORDER BY p.cycle_number, m.depth_level LIMIT 2000
                """
            df = pd.read_sql(query, engine)
            df = df.dropna(subset=['temperature', 'pressure', 'latitude', 'longitude'])
            return df
        except Exception:
            pass
    return generate_demo_data()

def get_float_list():
    if DB_AVAILABLE:
        try:
            import pandas as pd
            df = pd.read_sql("SELECT wmo_id FROM argo_floats ORDER BY id", engine)
            return df['wmo_id'].tolist()
        except Exception:
            pass
    return ['DEMO-1901771', 'DEMO-5906030', 'DEMO-4902915']

# ── VIEWS ─────────────────────────────────────────────────────────────────────
def index(request):
    floats = get_float_list()
    algorithms = [
        {'id': 'trajectory',  'label': '01 Trajectory Prediction',  'icon': '🛸'},
        {'id': 'ts_diagram',  'label': '02 T-S Diagram',             'icon': '🌡️'},
        {'id': 'temp_profile','label': '03 Temperature Profile',     'icon': '📈'},
        {'id': 'kmeans',      'label': '04 K-Means Clustering',      'icon': '🔵'},
        {'id': 'pca',         'label': '05 PCA Projection',          'icon': '🔷'},
        {'id': 'heatmap',     'label': '06 Density Heatmap',         'icon': '🔥'},
        {'id': 'knn',         'label': '07 kNN Nearest Neighbors',   'icon': '📡'},
        {'id': 'summary',     'label': '08 Float Summary Table',     'icon': '📊'},
        {'id': 'stats',       'label': '09 Temperature Statistics',  'icon': '📉'},
        {'id': 'profiles',    'label': '10 Top Profile Charts',      'icon': '🌊'},
    ]
    return render(request, 'dashboard/index.html', {
        'floats': floats,
        'algorithms': algorithms,
        'db_available': DB_AVAILABLE,
        'rag_available': RAG_AVAILABLE,
    })

def get_floats(request):
    return JsonResponse({'floats': get_float_list()})

@csrf_exempt
@require_POST
def ask_llm(request):
    body = json.loads(request.body)
    question = body.get('question', '').strip()
    if not question:
        return JsonResponse({'error': 'No question provided'}, status=400)

    if RAG_AVAILABLE and os.getenv('GROQ_API_KEY'):
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_collection(name='argo_profiles', embedding_function=EMBEDDING_FN)
            results = collection.query(query_texts=[question], n_results=5)
            chunks = results['documents'][0]
            context = '\n\n'.join([f"Profile {i+1}:\n{c}" for i, c in enumerate(chunks)])
            prompt = f"""You are an oceanography assistant.
Answer using ONLY the Argo float profile data below.
Provide exact numeric values — latitude, longitude, cycle number, date, temperature, salinity.
Do NOT guess. Only use what is in the data.

--- DATA ---
{context}
--- END ---

Question: {question}
Answer:"""
            response = groq_client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2, max_tokens=1024,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"[RAG Error: {e}] Demo answer: The selected Argo float shows temperature readings between 2°C–25°C across depths from surface to 2000 dbar."
    else:
        # Demo LLM response
        demo_answers = {
            'temperature': "Based on the profile data, surface temperatures range from 22–25°C, decreasing to 2–4°C at depths below 1000 dbar. The thermocline is clearly visible between 200–500 dbar.",
            'salinity': "Salinity values are stable at 35.0–35.2 PSU throughout the water column, with slight freshening near the surface due to precipitation effects.",
            'location': "The float is currently positioned at approximately 40°S, 65°E in the Southern Indian Ocean, tracking the Antarctic Circumpolar Current.",
            'cycle': "The most recent cycle shows an ascending profile with 120 valid measurement levels. Surface temperature was 18.3°C with salinity at 35.1 PSU.",
        }
        q_lower = question.lower()
        answer = next((v for k, v in demo_answers.items() if k in q_lower),
                      "This is a demo response. Connect your database and Groq API key to get real answers about the Argo float data. The system uses RAG (Retrieval-Augmented Generation) to search through embedded ocean profiles and answer your question with precise oceanographic data.")
    return JsonResponse({'answer': answer})

@csrf_exempt
@require_POST
def get_graph(request):
    body = json.loads(request.body)
    algo = body.get('algorithm', 'trajectory')
    float_wmo = body.get('float_wmo', None)
    k_clusters = int(body.get('k_clusters', 3))
    lat_input = float(body.get('lat_input', 0))
    lon_input = float(body.get('lon_input', 0))

    df = get_df(float_wmo)

    try:
        result = compute_algorithm(df, algo, k_clusters, lat_input, lon_input)
        result = clean_nan(result)
        from django.http import HttpResponse
        safe_json = json.dumps({'ok': True, 'data': result, 'algorithm': algo})
        return HttpResponse(safe_json, content_type='application/json')
    except Exception as e:
        safe_json = json.dumps({'ok': False, 'error': str(e)}, default=lambda x: None)
        return HttpResponse(safe_json, content_type='application/json', status=500)

def clean_nan(obj):
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif hasattr(obj, 'item'):  # numpy scalar
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    return obj

def compute_algorithm(df, algo, k_clusters=3, lat_input=0, lon_input=0):
    import pandas as pd

    if algo == 'trajectory':
        df['profile_date'] = pd.to_datetime(df['profile_date'])
        traj = df.groupby('profile_date')[['latitude','longitude']].mean().reset_index().dropna()
        if len(traj) < 2:
            return {'type': 'error', 'message': 'Not enough trajectory data'}
        t0 = traj['profile_date'].min()
        traj['days'] = (traj['profile_date'] - t0).dt.total_seconds() / 86400
        X = traj['days'].values.reshape(-1,1)
        lat_m = LinearRegression().fit(X, traj['latitude'])
        lon_m = LinearRegression().fit(X, traj['longitude'])
        fut = np.linspace(traj['days'].max()+1, traj['days'].max()+10, 10).reshape(-1,1)
        return {
            'type': 'trajectory',
            'observed_lat': traj['latitude'].tolist(),
            'observed_lon': traj['longitude'].tolist(),
            'future_lat': lat_m.predict(fut).tolist(),
            'future_lon': lon_m.predict(fut).tolist(),
            'dates': traj['profile_date'].dt.strftime('%Y-%m-%d').tolist(),
        }

    elif algo == 'ts_diagram':
        sample = df[['temperature','pressure','salinity','cycle_number']].dropna().head(500)
        return {
            'type': 'scatter',
            'x': sample['temperature'].tolist(),
            'y': sample['pressure'].tolist(),
            'color': sample['cycle_number'].tolist(),
            'xlabel': 'Temperature (°C)', 'ylabel': 'Pressure (dbar)',
            'title': 'T-S Diagram', 'invert_y': True,
        }

    elif algo == 'temp_profile':
        fit = df[['pressure','temperature']].dropna()
        if len(fit) < 2:
            return {'type': 'error', 'message': 'Not enough data'}
        X = fit[['pressure']]; y = fit['temperature']
        model = LinearRegression().fit(X, y)
        px_range = np.linspace(X['pressure'].min(), X['pressure'].max(), 80).reshape(-1,1)
        return {
            'type': 'profile',
            'actual_temp': y.tolist(),
            'actual_pres': X['pressure'].tolist(),
            'pred_temp': model.predict(px_range).tolist(),
            'pred_pres': px_range.flatten().tolist(),
            'r2': round(model.score(X, y), 3),
        }

    elif algo == 'kmeans':
        coords = df[['latitude','longitude']].dropna()
        km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(coords)
        centers = km.cluster_centers_
        return {
            'type': 'map_cluster',
            'lats': coords['latitude'].tolist(),
            'lons': coords['longitude'].tolist(),
            'labels': labels.tolist(),
            'center_lats': centers[:,0].tolist(),
            'center_lons': centers[:,1].tolist(),
        }

    elif algo == 'pca':
        coords = df[['latitude','longitude']].dropna()
        scaled = StandardScaler().fit_transform(coords)
        pca = PCA(n_components=2)
        comp = pca.fit_transform(scaled)
        explained = pca.explained_variance_ratio_
        return {
            'type': 'scatter2d',
            'x': comp[:,0].tolist(), 'y': comp[:,1].tolist(),
            'xlabel': f'PC1 ({explained[0]*100:.1f}%)',
            'ylabel': f'PC2 ({explained[1]*100:.1f}%)',
            'title': 'PCA Projection of Float Positions',
        }

    elif algo == 'heatmap':
        df2 = df.copy()
        df2['lat_bin'] = df2['latitude'].round()
        df2['lon_bin'] = df2['longitude'].round()
        heat = df2.groupby(['lat_bin','lon_bin']).size().reset_index(name='count')
        return {
            'type': 'heatmap',
            'lats': heat['lat_bin'].tolist(),
            'lons': heat['lon_bin'].tolist(),
            'counts': heat['count'].tolist(),
        }

    elif algo == 'knn':
        coords = df[['latitude','longitude']].dropna()
        nn = NearestNeighbors(n_neighbors=min(5, len(coords)))
        nn.fit(coords)
        query = [[lat_input or coords['latitude'].mean(), lon_input or coords['longitude'].mean()]]
        dist, idx = nn.kneighbors(query)
        neighbors = df.iloc[idx[0]][['cycle_number','latitude','longitude','temperature','pressure']].reset_index(drop=True)
        return {
            'type': 'table',
            'title': f'5 Nearest Floats to ({query[0][0]:.3f}, {query[0][1]:.3f})',
            'columns': list(neighbors.columns),
            'rows': neighbors.round(3).values.tolist(),
            'distances': dist[0].round(4).tolist(),
        }

    elif algo == 'summary':
        df_clean = df.copy()
        df_clean['salinity'] = df_clean['salinity'].replace([np.inf, -np.inf], np.nan)
        df_clean['temperature'] = df_clean['temperature'].replace([np.inf, -np.inf], np.nan)
        df_clean['pressure'] = df_clean['pressure'].replace([np.inf, -np.inf], np.nan)
        summary = df_clean.groupby('cycle_number')[['temperature','pressure','salinity']].agg(['min','max','mean']).reset_index()
        summary.columns = ['Cycle', 'Temp Min', 'Temp Max', 'Temp Mean', 'Pres Min', 'Pres Max', 'Pres Mean', 'Sal Min', 'Sal Max', 'Sal Mean']
        summary = summary.round(3)
        return {
            'type': 'table',
            'title': 'Float Summary by Cycle',
            'columns': list(summary.columns),
            'rows': summary.round(3).values.tolist(),}

    elif algo == 'stats':
        stats = df['temperature'].describe()
        return {
            'type': 'stats',
            'labels': list(stats.index),
            'values': [round(v, 4) for v in stats.values],
            'histogram_x': df['temperature'].dropna().tolist(),
        }

    elif algo == 'profiles':
        top3 = sorted(df['cycle_number'].unique())[:3]
        series = []
        for cyc in top3:
            sub = df[df['cycle_number'] == cyc].sort_values('pressure')
            sub = sub[['temperature','pressure']].dropna()
            series.append({
                'name': f'Cycle {cyc}',
                'temp': sub['temperature'].tolist(),
                'pressure': sub['pressure'].tolist(),
            })
        return {'type': 'multi_profile', 'series': series}
