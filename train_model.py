"""
=============================================================
  Sentinel Hub — Spatial-Temporal Fusion (STF) Model
  train_model.py
=============================================================

  Architecture: Two-branch fusion model
  ┌──────────────────────────────────────────────────────┐
  │  TEMPORAL BRANCH                                     │
  │  Fill-level time series (24h sliding window)         │
  │  → Statistical: mean, std, slope, peaks, IQR         │
  │  → Rate-of-change over 1h / 3h / 6h windows         │
  │  → Hour-of-day & day-of-week cyclical encoding       │
  ├──────────────────────────────────────────────────────┤
  │  SPATIAL BRANCH                                      │
  │  Bin location (lat/lng), sector, waste type          │
  │  → K=3 nearest-neighbour fills & distances           │
  │  → Bins within 2km density index                     │
  │  → Distance to depot, sector average fill            │
  ├──────────────────────────────────────────────────────┤
  │  FUSION LAYER                                        │
  │  Concatenate spatial + temporal → StandardScaler     │
  │  → GradientBoostingClassifier (300 trees)            │
  │  → Predicts: Low / Medium / High / Critical          │
  └──────────────────────────────────────────────────────┘

  Run:
      pip install numpy scikit-learn matplotlib joblib
      python train_model.py
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib

np.random.seed(42)

# ─── Bin registry (matches dashboard) ───────────────────────
BINS_META = [
    {'id':'B-001','lat':13.0827,'lng':80.2707,'type':'Mixed General', 'sector':'Alpha'},
    {'id':'B-002','lat':13.0500,'lng':80.2824,'type':'Mixed Waste',   'sector':'Beta'},
    {'id':'B-003','lat':13.0732,'lng':80.2609,'type':'Recyclable',    'sector':'Alpha'},
    {'id':'B-004','lat':13.0418,'lng':80.2341,'type':'Organic',       'sector':'Gamma'},
    {'id':'B-005','lat':13.0569,'lng':80.2521,'type':'General',       'sector':'Beta'},
    {'id':'B-006','lat':12.9784,'lng':80.2209,'type':'Organic',       'sector':'Delta'},
    {'id':'B-007','lat':13.0067,'lng':80.2563,'type':'Recyclable',    'sector':'Gamma'},
    {'id':'B-008','lat':13.0604,'lng':80.2428,'type':'General',       'sector':'Alpha'},
    {'id':'B-009','lat':13.0359,'lng':80.1574,'type':'Mixed Waste',   'sector':'Epsilon'},
    {'id':'B-010','lat':13.0067,'lng':80.2197,'type':'Hazardous',     'sector':'Delta'},
    {'id':'B-011','lat':12.9010,'lng':80.2279,'type':'Recyclable',    'sector':'Zeta'},
    {'id':'B-012','lat':12.9249,'lng':80.1000,'type':'General',       'sector':'Zeta'},
]
DEPOT_LAT, DEPOT_LNG = 13.0950, 80.2350
WASTE_TYPES = ['Mixed General','Mixed Waste','Recyclable','Organic','General','Hazardous']
SECTORS     = ['Alpha','Beta','Gamma','Delta','Epsilon','Zeta']
LABEL_NAMES = ['Low','Medium','High','Critical']
THRESHOLDS  = {'Low':(0,39),'Medium':(40,69),'High':(70,89),'Critical':(90,100)}


def haversine_km(la1,ln1,la2,ln2):
    R=6371.0; dl=np.radians(la2-la1); dn=np.radians(ln2-ln1)
    a=np.sin(dl/2)**2+np.cos(np.radians(la1))*np.cos(np.radians(la2))*np.sin(dn/2)**2
    return R*2*np.arcsin(np.sqrt(a))


# ═════════════════════════════════════════════════════════════
#  TEMPORAL BRANCH
#  Simulate 24h fill time-series, extract features
# ═════════════════════════════════════════════════════════════

def generate_series(fill_pct, waste_type, sector, sph=4):
    """Synthetic 24h fill time-series ending at fill_pct."""
    n = 24 * sph
    t = np.linspace(0, 24, n)
    type_rate = {'Organic':1.4,'Mixed Waste':1.3,'Mixed General':1.2,
                 'General':1.0,'Recyclable':0.9,'Hazardous':0.6}.get(waste_type,1.0)
    sec_mult  = {'Alpha':1.3,'Beta':1.2,'Gamma':1.1,'Delta':1.0,'Epsilon':0.9,'Zeta':0.8}.get(sector,1.0)
    diurnal   = (1.8*np.exp(-0.5*((t-8.0)/1.2)**2) +
                 1.2*np.exp(-0.5*((t-13.0)/1.0)**2) +
                 1.6*np.exp(-0.5*((t-19.5)/1.5)**2) + 0.4)
    diurnal  /= diurnal.max()
    incs      = type_rate*sec_mult*diurnal*(fill_pct/(diurnal.sum()+1e-9))
    incs     += np.random.normal(0, fill_pct*0.02, n)
    s         = np.cumsum(np.clip(incs,0,None))
    if s[-1]>0: s = s*(fill_pct/s[-1])
    return np.clip(s,0,100).astype(np.float32)


def temporal_features(series, hour=None):
    """Extract temporal feature vector from fill series."""
    n = len(series)
    f = []
    # Statistical
    f += [np.mean(series),np.std(series),np.min(series),np.max(series),
          np.median(series),float(np.percentile(series,75)-np.percentile(series,25)),
          float(np.percentile(series,90)),series[-1]]
    # Trend
    x = np.arange(n)
    slope = np.polyfit(x,series,1)[0]
    f.append(float(slope*n))
    # Short-term slopes
    for w in [4,12,24,48]:
        ww=min(w,n)
        f.append(float(np.polyfit(np.arange(ww),series[-ww:],1)[0]*ww) if ww>1 else 0.0)
    # Differences
    d1=np.diff(series); d2=np.diff(d1)
    f+=[float(np.mean(d1)),float(np.std(d1)),float(np.max(d1)),float(d1[-1]),
        float(np.mean(d2)) if len(d2)>0 else 0.0]
    # Rate of change
    for w in [4,12,24]:
        f.append(float(series[-1]-series[-min(w+1,n)]))
    # Peaks
    above = np.where(series>75)[0]
    f += [float(len(above)), float(n-above[-1]) if len(above)>0 else float(n)]
    # Cyclical time
    if hour is None: hour=np.random.uniform(0,24)
    dow = np.random.randint(0,7)
    f += [np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
          np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)]
    return np.array(f,dtype=np.float32)


# ═════════════════════════════════════════════════════════════
#  SPATIAL BRANCH
# ═════════════════════════════════════════════════════════════

def spatial_features(bin_meta, all_bins_fills):
    """Extract spatial feature vector for one bin."""
    f = []
    # Normalised position
    f += [(bin_meta['lat']-13.0350)*111.0, (bin_meta['lng']-80.2100)*91.0]
    # Depot distance
    f.append(haversine_km(bin_meta['lat'],bin_meta['lng'],DEPOT_LAT,DEPOT_LNG))
    # Waste type one-hot
    f += [1.0 if bin_meta['type']==wt else 0.0 for wt in WASTE_TYPES]
    # Sector one-hot
    f += [1.0 if bin_meta['sector']==s else 0.0 for s in SECTORS]
    # Nearest neighbours
    dists=sorted([(haversine_km(bin_meta['lat'],bin_meta['lng'],o['lat'],o['lng']),o['fill'])
                  for o in all_bins_fills if o['id']!=bin_meta['id']])
    for i in range(3):
        f += list(dists[i]) if i<len(dists) else [0.0,0.0]
    # Density + max/mean within radius
    f2km=[fill for d,fill in dists if d<=2.0]
    f3km=[fill for d,fill in dists if d<=3.0]
    f += [float(len(f2km)),
          float(max(f3km)) if f3km else 0.0,
          float(np.mean(f3km)) if f3km else 0.0]
    # Sector average
    sa=[o['fill'] for o in all_bins_fills if o['sector']==bin_meta['sector'] and o['id']!=bin_meta['id']]
    f.append(float(np.mean(sa)) if sa else 0.0)
    return np.array(f,dtype=np.float32)


def fused_feature(bin_meta, fill_pct, all_fills, hour=None):
    series = generate_series(fill_pct, bin_meta['type'], bin_meta['sector'])
    t = temporal_features(series, hour)
    s = spatial_features(bin_meta, all_fills)
    return np.concatenate([t,s])


def fill_to_label(fill):
    for lbl,(lo,hi) in THRESHOLDS.items():
        if lo<=fill<=hi: return lbl
    return 'Critical'


# ═════════════════════════════════════════════════════════════
#  GENERATE DATASET
# ═════════════════════════════════════════════════════════════

print("="*60)
print("  Sentinel Hub — Spatial-Temporal Fusion (STF) Training")
print("="*60)
print("\n[1/5] Generating dataset...")

FILLS = [5,12,20,28,35, 42,50,57,63,68, 72,78,83,87, 91,94,97,99]
N_VAR = 70

X_list, y_list = [], []
for bm in BINS_META:
    for fill in FILLS:
        label = fill_to_label(fill)
        for _ in range(N_VAR):
            all_fills=[{**b,'fill':float(np.clip(fill+np.random.normal(0,18),0,100))} for b in BINS_META]
            hour = float(np.random.choice(range(0,24,3))) + np.random.uniform(0,3)
            X_list.append(fused_feature(bm,fill,all_fills,hour))
            y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)

# Compute branch sizes
t_sz = len(temporal_features(np.zeros(96)))
s_sz = len(spatial_features(BINS_META[0],[{**b,'fill':50.0} for b in BINS_META]))
print(f"  Samples        : {len(X)}")
print(f"  Feature vector : {X.shape[1]}d  (temporal={t_sz}, spatial={s_sz})")
print(f"  Classes        : { {k:int((y==k).sum()) for k in LABEL_NAMES} }")


# ═════════════════════════════════════════════════════════════
#  TRAIN
# ═════════════════════════════════════════════════════════════

print("\n[2/5] Encoding labels...")
le = LabelEncoder(); le.fit(LABEL_NAMES); y_enc = le.transform(y)

print("[3/5] Training STF GradientBoosting model...")
X_tr,X_te,y_tr,y_te = train_test_split(X,y_enc,test_size=0.2,random_state=42,stratify=y_enc)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbm', GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        min_samples_split=10, min_samples_leaf=4,
        subsample=0.85, max_features='sqrt', random_state=42,
        validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
    )),
])
model.fit(X_tr,y_tr)

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_sc = cross_val_score(model,X,y_enc,cv=cv,scoring='accuracy',n_jobs=-1)
print(f"  CV  accuracy: {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
print(f"  Test accuracy: {model.score(X_te,y_te):.4f}")
y_pred = model.predict(X_te)
print("\n  Classification Report:")
print(classification_report(y_te,y_pred,target_names=le.classes_))


# ═════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE PLOT
# ═════════════════════════════════════════════════════════════

print("[4/5] Saving feature importance plot...")
t_names = (['mean','std','min','max','median','IQR','p90','current','slope_full']+
           [f'slope_{w}h' for w in ['1','3','6','12']]+
           ['d1_mean','d1_std','d1_max','d1_last','d2_mean']+
           ['delta_1h','delta_3h','delta_6h','peaks','time_peak',
            'sin_h','cos_h','sin_dow','cos_dow'])
s_names = (['north','east','depot_dist']+
           [f'type_{w[:4]}' for w in WASTE_TYPES]+
           [f'sec_{s}' for s in SECTORS]+
           [f'knn{i+1}d' for i in range(3)]+[f'knn{i+1}f' for i in range(3)]+
           ['dens_2km','max_3km','mean_3km','sec_avg'])
all_fnames = t_names+s_names

imp = model.named_steps['gbm'].feature_importances_
top = np.argsort(imp)[-20:][::-1]
fig,ax=plt.subplots(figsize=(10,6),facecolor='#0d1117')
ax.set_facecolor('#0d1117')
cols=['#60a5fa' if i<t_sz else '#a78bfa' for i in top]
ax.barh(range(20),imp[top][::-1],color=cols[::-1],alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels([all_fnames[i] if i<len(all_fnames) else f'f{i}' for i in top][::-1],
                   color='#e5e7eb',fontsize=9)
ax.set_xlabel('Importance',color='#9ca3af'); ax.set_title('STF Model — Feature Importances\nBlue=Temporal  Purple=Spatial',color='white',fontsize=11)
ax.tick_params(colors='#9ca3af')
for sp in ax.spines.values(): sp.set_edgecolor('#374151')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc='#60a5fa',label='Temporal'),Patch(fc='#a78bfa',label='Spatial')],
          loc='lower right',facecolor='#1f2937',labelcolor='white',fontsize=9)
plt.tight_layout(); plt.savefig('stf_feature_plot.png',dpi=150,bbox_inches='tight',facecolor='#0d1117'); plt.close()
print("  Saved: stf_feature_plot.png")


# ═════════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════

print("\n[5/5] Saving model artifacts...")
joblib.dump(model,       'stf_model.pkl')
joblib.dump(le,          'stf_encoder.pkl')
joblib.dump(all_fnames,  'stf_feature_names.pkl')
print("  stf_model.pkl  stf_encoder.pkl  stf_feature_names.pkl")
print()
print("="*60)
print(f"  STF model trained!  CV={cv_sc.mean():.1%}")
print(f"  Branches: Temporal {t_sz}d + Spatial {s_sz}d = {t_sz+s_sz}d fused")
print("  Next: python app.py")
print("="*60)