"""
=============================================================
  Sentinel Hub — Flask API  (Spatial-Temporal Fusion)
  app.py
=============================================================
  HOW TO USE:
  1. Run:  python app.py
  2. Open: sentinel-hub-ml.html  directly in your browser
           (double-click the file, or drag it into Chrome/Firefox)
           DO NOT navigate to http://localhost:5000 for the UI.

  The API runs at http://localhost:5000 and the dashboard
  HTML file fetches predictions from it automatically.
=============================================================
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import joblib
import os, time, random

app = Flask(__name__)
CORS(app)

# ── Load STF model ───────────────────────────────────────────
try:
    MODEL   = joblib.load('stf_model.pkl')
    ENCODER = joblib.load('stf_encoder.pkl')
    print("[✓] STF model loaded")
except FileNotFoundError:
    MODEL = ENCODER = None
    print("[!] Run python train_model.py first")

# ── Constants (must match train_model.py) ────────────────────
DEPOT_LAT, DEPOT_LNG = 13.0950, 80.2350
WASTE_TYPES = ['Mixed General','Mixed Waste','Recyclable','Organic','General','Hazardous']
SECTORS     = ['Alpha','Beta','Gamma','Delta','Epsilon','Zeta']

BINS = [
    {'id':'B-001','name':'Central Market',     'lat':13.0827,'lng':80.2707,'fill':94,'type':'Mixed General', 'last':'04:30 AM','truck':'T-01','sector':'Alpha'},
    {'id':'B-002','name':'Marina Beach North', 'lat':13.0500,'lng':80.2824,'fill':91,'type':'Mixed Waste',   'last':'06:00 AM','truck':'T-02','sector':'Beta'},
    {'id':'B-003','name':'Egmore Station',     'lat':13.0732,'lng':80.2609,'fill':87,'type':'Recyclable',    'last':'04:45 AM','truck':'T-01','sector':'Alpha'},
    {'id':'B-004','name':'T.Nagar Hub',        'lat':13.0418,'lng':80.2341,'fill':83,'type':'Organic',       'last':'06:30 AM','truck':'T-02','sector':'Gamma'},
    {'id':'B-005','name':'Anna Salai',         'lat':13.0569,'lng':80.2521,'fill':78,'type':'General',       'last':'07:00 AM','truck':'T-01','sector':'Beta'},
    {'id':'B-006','name':'Velachery Market',   'lat':12.9784,'lng':80.2209,'fill':96,'type':'Organic',       'last':'04:00 AM','truck':'T-03','sector':'Delta'},
    {'id':'B-007','name':'Adyar Junction',     'lat':13.0067,'lng':80.2563,'fill':85,'type':'Recyclable',    'last':'07:15 AM','truck':'T-02','sector':'Gamma'},
    {'id':'B-008','name':'Nungambakkam',       'lat':13.0604,'lng':80.2428,'fill':74,'type':'General',       'last':'07:30 AM','truck':'T-01','sector':'Alpha'},
    {'id':'B-009','name':'Porur Junction',     'lat':13.0359,'lng':80.1574,'fill':90,'type':'Mixed Waste',   'last':'05:45 AM','truck':'T-04','sector':'Epsilon'},
    {'id':'B-010','name':'Guindy Industrial',  'lat':13.0067,'lng':80.2197,'fill':98,'type':'Hazardous',     'last':'05:00 AM','truck':'T-03','sector':'Delta'},
    {'id':'B-011','name':'Sholinganallur',     'lat':12.9010,'lng':80.2279,'fill':81,'type':'Recyclable',    'last':'08:30 AM','truck':'T-04','sector':'Zeta'},
    {'id':'B-012','name':'Tambaram',           'lat':12.9249,'lng':80.1000,'fill':88,'type':'General',       'last':'08:00 AM','truck':'T-04','sector':'Zeta'},
]
BIN_MAP = {b['id']: b for b in BINS}
_fill_drift = {}

def live_fill(bid):
    if bid not in _fill_drift: _fill_drift[bid] = 0.0
    _fill_drift[bid] += random.uniform(-0.2, 0.4)
    _fill_drift[bid]  = max(-4, min(4, _fill_drift[bid]))
    return round(min(100, max(0, BIN_MAP[bid]['fill'] + _fill_drift[bid])), 1)


# ── Feature builders (mirror train_model.py) ─────────────────
def haversine_km(la1,ln1,la2,ln2):
    R=6371.0; dl=np.radians(la2-la1); dn=np.radians(ln2-ln1)
    a=np.sin(dl/2)**2+np.cos(np.radians(la1))*np.cos(np.radians(la2))*np.sin(dn/2)**2
    return R*2*np.arcsin(np.sqrt(a))

def generate_series(fill_pct, waste_type, sector, sph=4):
    n=24*sph; t=np.linspace(0,24,n)
    tr={'Organic':1.4,'Mixed Waste':1.3,'Mixed General':1.2,'General':1.0,'Recyclable':0.9,'Hazardous':0.6}.get(waste_type,1.0)
    sm={'Alpha':1.3,'Beta':1.2,'Gamma':1.1,'Delta':1.0,'Epsilon':0.9,'Zeta':0.8}.get(sector,1.0)
    d=(1.8*np.exp(-0.5*((t-8.0)/1.2)**2)+1.2*np.exp(-0.5*((t-13.0)/1.0)**2)+1.6*np.exp(-0.5*((t-19.5)/1.5)**2)+0.4)
    d/=d.max()
    inc=tr*sm*d*(fill_pct/(d.sum()+1e-9))+np.random.normal(0,fill_pct*0.02,n)
    s=np.cumsum(np.clip(inc,0,None))
    if s[-1]>0: s=s*(fill_pct/s[-1])
    return np.clip(s,0,100).astype(np.float32)

def temporal_features(series, hour=None):
    n=len(series); f=[]
    f+=[float(np.mean(series)),float(np.std(series)),float(np.min(series)),float(np.max(series)),
        float(np.median(series)),float(np.percentile(series,75)-np.percentile(series,25)),
        float(np.percentile(series,90)),float(series[-1])]
    slope=np.polyfit(np.arange(n),series,1)[0]; f.append(float(slope*n))
    for w in [4,12,24,48]:
        ww=min(w,n); f.append(float(np.polyfit(np.arange(ww),series[-ww:],1)[0]*ww) if ww>1 else 0.0)
    d1=np.diff(series); d2=np.diff(d1)
    f+=[float(np.mean(d1)),float(np.std(d1)),float(np.max(d1)),float(d1[-1]),
        float(np.mean(d2)) if len(d2)>0 else 0.0]
    for w in [4,12,24]: f.append(float(series[-1]-series[-min(w+1,n)]))
    above=np.where(series>75)[0]
    f+=[float(len(above)),float(n-above[-1]) if len(above)>0 else float(n)]
    if hour is None: hour=float(time.localtime().tm_hour)+float(time.localtime().tm_min)/60
    dow=time.localtime().tm_wday
    f+=[np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)]
    return np.array(f,dtype=np.float32)

def spatial_features(bm, all_bins_fills):
    f=[(bm['lat']-13.0350)*111.0,(bm['lng']-80.2100)*91.0,
       haversine_km(bm['lat'],bm['lng'],DEPOT_LAT,DEPOT_LNG)]
    f+=[1.0 if bm['type']==wt else 0.0 for wt in WASTE_TYPES]
    f+=[1.0 if bm['sector']==s else 0.0 for s in SECTORS]
    dists=sorted([(haversine_km(bm['lat'],bm['lng'],o['lat'],o['lng']),o['fill'])
                  for o in all_bins_fills if o['id']!=bm['id']])
    for i in range(3): f+=list(dists[i]) if i<len(dists) else [0.0,0.0]
    f2=[fl for d,fl in dists if d<=2.0]; f3=[fl for d,fl in dists if d<=3.0]
    f+=[float(len(f2)),float(max(f3)) if f3 else 0.0,float(np.mean(f3)) if f3 else 0.0]
    sa=[o['fill'] for o in all_bins_fills if o.get('sector')==bm['sector'] and o['id']!=bm['id']]
    f.append(float(np.mean(sa)) if sa else 0.0)
    return np.array(f,dtype=np.float32)

def rule_predict(fill):
    """Fallback when model not loaded."""
    if fill>=90:  label,conf='Critical',0.92
    elif fill>=70:label,conf='High',0.88
    elif fill>=40:label,conf='Medium',0.84
    else:         label,conf='Low',0.90
    p={k:0.01 for k in ['Low','Medium','High','Critical']}; p[label]=conf
    return {'label':label,'confidence':conf,'probabilities':p,'method':'rule-based (model not loaded)'}

def stf_predict(bin_id, fill_override=None):
    b = BIN_MAP.get(bin_id)
    if not b: return None
    fill = fill_override if fill_override is not None else live_fill(bin_id)
    if MODEL is None: return {**rule_predict(fill), 'live_fill': fill}

    # Build all-bins context with current live fills
    all_ctx = [{**BIN_MAP[bid], 'fill': live_fill(bid)} for bid in BIN_MAP]

    s   = generate_series(fill, b['type'], b['sector'])
    t_f = temporal_features(s)
    s_f = spatial_features(b, all_ctx)
    feat= np.concatenate([t_f, s_f]).reshape(1,-1)

    proba = MODEL.predict_proba(feat)[0]
    idx   = np.argmax(proba)
    label = ENCODER.classes_[idx]
    probs = {cls: float(p) for cls,p in zip(ENCODER.classes_, proba)}
    return {'label':label,'confidence':float(proba[idx]),'probabilities':probs,
            'method':'Spatial-Temporal Fusion (STF)', 'live_fill': fill}


# ── Routes ────────────────────────────────────────────────────
@app.route('/api/status')
def status():
    return jsonify({'status':'online','model':'Spatial-Temporal Fusion (STF)',
                    'model_loaded': MODEL is not None,'bins':len(BINS)})

@app.route('/api/bins')
def get_bins():
    result=[]
    for b in BINS:
        pred=stf_predict(b['id'])
        result.append({**b,'live_fill':pred['live_fill'],'prediction':pred})
    return jsonify(result)

@app.route('/api/predict/<bin_id>')
def predict_bin(bin_id):
    pred=stf_predict(bin_id)
    if not pred: return jsonify({'error':'Not found'}),404
    b=BIN_MAP[bin_id]
    return jsonify({'bin_id':bin_id,'name':b['name'],'prediction':pred,'timestamp':time.time()})

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    data=request.get_json(silent=True) or {}
    ids=data.get('ids') or list(BIN_MAP.keys())
    return jsonify([{'bin_id':bid,'prediction':stf_predict(bid)} for bid in ids if bid in BIN_MAP])

@app.route('/')
def index():
    # Try to serve the HTML from several common locations
    candidates = [
        'sentinel-hub-ml.html',
        os.path.join(os.path.dirname(__file__), 'sentinel-hub-ml.html'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return send_file(os.path.abspath(p))
    # Fallback: redirect instructions page
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>Sentinel Hub API</title>
  <style>
    body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;
         display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}
    .card{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:40px 48px;max-width:500px;text-align:center;}
    h1{color:#60a5fa;font-size:22px;margin-bottom:8px;}
    p{color:#94a3b8;font-size:14px;line-height:1.6;}
    code{background:#0f172a;padding:3px 8px;border-radius:5px;color:#a3f69c;font-size:13px;}
    .tick{font-size:36px;margin-bottom:16px;}
    a{color:#60a5fa;text-decoration:none;}
  </style>
</head>
<body>
  <div class="card">
    <div class="tick">✅</div>
    <h1>Sentinel Hub STF API is running</h1>
    <p>The API is online at <code>http://localhost:5000</code></p>
    <p style="margin-top:16px;">
      To use the dashboard, simply open<br/>
      <code>sentinel-hub-ml.html</code><br/>
      directly in your browser — <strong>do not open it through this URL</strong>.
    </p>
    <p style="margin-top:16px;">
      API endpoints:<br/>
      <a href="/api/status">/api/status</a> &nbsp;·&nbsp;
      <a href="/api/bins">/api/bins</a>
    </p>
  </div>
</body>
</html>"""

if __name__=='__main__':
    print("="*50)
    print("  Sentinel Hub STF API — http://localhost:5000")
    print("  Model: Spatial-Temporal Fusion (GBM)")
    print("="*50)
    app.run(debug=True,port=5000,host='0.0.0.0')