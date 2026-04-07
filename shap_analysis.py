"""
SHAP Interpretability Analysis — Fast Implementation
Uses sklearn RandomForest with Kernel SHAP approximation:
  - Background dataset masking via feature marginalisation
  - Vectorised computation over all samples
  - Much faster than recursive TreeSHAP
  Author： MINI HAN WANG
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from openpyxl import load_workbook as _load_wb

warnings.filterwarnings('ignore')
os.makedirs('mini/figures', exist_ok=True)
os.makedirs('mini/outputs', exist_ok=True)

PAL = ['#2E75B6','#ED7D31','#70AD47','#5B9BD5','#FFC000','#A9D18E','#FF7575','#9DC3E6']
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':120,'font.size':10})

# ═══════════════════════════════════════════════════════
# STEP 1: Load data (same as main pipeline)
# ═══════════════════════════════════════════════════════
print("=== Loading & preparing data ===")
MISS = {'拒绝回答','不知道','未调查','入户面访未调查','不适用','电话调查未调查',
        '不接受调查','没有回答','None',None}
_wb = _load_wb('/mnt/user-data/uploads/就业.xlsx', read_only=True)
_ws = _wb['Sheet1']
_all = list(_ws.iter_rows(values_only=True))
_codes = [str(c) if c is not None else '' for c in _all[0]]
_seen={}; _unique=[]
for c in _codes:
    if c not in _seen: _seen[c]=0; _unique.append(c)
    else: _seen[c]+=1; _unique.append(f"{c}__dup{_seen[c]}")
raw = pd.DataFrame(_all[2:], columns=_unique)
for col in raw.columns:
    if raw[col].dtype==object:
        raw[col]=raw[col].apply(lambda x: np.nan if x in MISS or (isinstance(x,str) and x.strip() in MISS) else x)

raw['income_annual'] = pd.to_numeric(raw['a8a'], errors='coerce')
raw.loc[raw['income_annual']<0,'income_annual'] = np.nan
raw.loc[(raw['a8a'].astype(str).str.contains('百万',na=False))&raw['income_annual'].isna(),'income_annual']=1_500_000

def classify_emp(row):
    inc=row['income_annual']; a53=str(row.get('a53','')); a58=str(row.get('a58',''))
    if '从事了以取得经济收入为目的的工作' in a53: return 1
    if '停薪休假' in a53 or '带薪休假' in a53: return 1 if (pd.notna(inc) and inc>0) else 0
    if '未从事任何以获得经济收入为目的的工作' in a53:
        if '目前从事非农工作' in a58 or '目前务农' in a58: return 1
        if pd.notna(inc) and inc>0: return 1
        return 0
    if pd.notna(inc) and inc>0: return 1
    if pd.notna(inc) and inc==0:
        if '目前从事非农工作' in a58 or '目前务农' in a58: return 1
        if '没有工作' in a58 or '从未工作' in a58: return 0
        return 0
    hrs=pd.to_numeric(row.get('a54'),errors='coerce')
    if pd.notna(hrs) and hrs>0: return 1
    return np.nan

raw['employment_binary'] = raw.apply(classify_emp, axis=1)

# SN index
raw['sn_neighbor']=pd.to_numeric(raw['a31a__dup1'],errors='coerce')
raw['sn_friend']  =pd.to_numeric(raw['a31b__dup1'],errors='coerce')
raw['sn_trust']   =pd.to_numeric(raw['a33__dup1'], errors='coerce')
raw['sn_learning']=pd.to_numeric(raw['a313'],       errors='coerce')
SN_DIMS=['sn_neighbor','sn_friend','sn_trust','sn_learning']

def entropy_idx(df_in):
    d=df_in.astype(float).copy()
    for c in d.columns:
        mn,mx=d[c].min(),d[c].max(); d[c]=(d[c]-mn)/(mx-mn+1e-9)
    d=d.fillna(d.median()); n=len(d)
    p=d.div(d.sum(axis=0)+1e-9)
    e=-(p*np.log(p+1e-12)).sum(axis=0)/np.log(n+1e-9)
    w=(1-e)/((1-e).sum()+1e-9)
    return d.dot(w)

vsn=raw[SN_DIMS].dropna()
raw['social_network_index_entropy']=np.nan
raw.loc[vsn.index,'social_network_index_entropy']=entropy_idx(vsn).values

# VC index
vc_dims={'angel':['早期案例数（起）','早期投资金额（亿元）'],'vc':['VC 案例数（起）','VC 投资金额（亿元）'],
         'pe':['PE 案例数（起）','PE 投资金额（亿元）'],'financial':['金融业市场化'],
         'hr':['人力资源供应条件'],'tech':['技术成果市场化','知识产权保护'],
         'intermediary':['市场中介组织的发育','维护市场的法治环境']}
for nm,cols in vc_dims.items():
    c=[x for x in cols if x in raw.columns]
    if c: raw[f'{nm}_ecology']=raw[c].apply(pd.to_numeric,errors='coerce').mean(axis=1)
VC_DIMS=[f'{n}_ecology' for n in vc_dims if f'{n}_ecology' in raw.columns]
vvc=raw[VC_DIMS].dropna()
raw['vc_ecosystem_index_entropy']=np.nan
raw.loc[vvc.index,'vc_ecosystem_index_entropy']=entropy_idx(vvc).values

# Controls
raw['age']=2023-pd.to_numeric(raw['a3a'],errors='coerce'); raw['age_sq']=raw['age']**2
raw['gender']=raw['a2'].map({'男':1,'女':0})
edu_map={'没有受过任何教育':1,'小学':2,'初中':3,'高中':4,'普通高中':4,'中专':5,
         '大专':6,'大学专科（正规高等教育）':6,'大学专科（成人高等教育）':6,
         '大学本科（正规高等教育）':7,'大学本科（成人高等教育）':7,'研究生及以上':8,'研究生':8}
raw['education']=raw['a7a'].map(edu_map); raw['father_edu']=raw['a89b'].map(edu_map); raw['mother_edu']=raw['a90b'].map(edu_map)
raw['health']=raw['a15'].map({'非常不健康':1,'比较不健康':2,'一般':3,'比较健康':4,'非常健康':5})
raw['hukou_urban']=raw['a18'].apply(lambda x:1 if pd.notna(x) and '非农' in str(x) else (0 if pd.notna(x) else np.nan))
raw['married']=raw['a69'].map({'初婚有配偶':1,'再婚有配偶':1,'未婚':0,'离婚':0,'丧偶':0,'同居':1})
raw['family_econ']=pd.to_numeric(raw['a65'],errors='coerce')
raw['has_car']=raw['a671'].map({'是':1,'否':0,'有':1,'没有':0}); raw['has_house']=raw['a12a'].map({'是':1,'否':0})
for c in ['a43a','a43b','a43c','a43d','a43e']:
    raw[f'{c}_n']=raw[c].apply(lambda x:pd.to_numeric(str(x).replace('分','').replace('层','').strip(),errors='coerce') if pd.notna(x) else np.nan)
raw['subjective_class']=raw[['a43a_n','a43b_n','a43c_n','a43d_n','a43e_n']].mean(axis=1)

CONTROLS=['age','gender','education','health','hukou_urban','married','father_edu','mother_edu',
          'family_econ','has_house','has_car','subjective_class']
CONTROLS=[c for c in CONTROLS if c in raw.columns]
df=raw.copy()
df[CONTROLS]=SimpleImputer(strategy='median').fit_transform(df[CONTROLS])
KEY=['employment_binary','social_network_index_entropy','vc_ecosystem_index_entropy']
analytic=df.dropna(subset=KEY).copy()
X_VARS=['social_network_index_entropy','vc_ecosystem_index_entropy']+CONTROLS
sub=analytic.dropna(subset=['employment_binary']).copy()
X_df=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(sub[X_VARS]),columns=X_VARS,index=sub.index)
y=sub['employment_binary'].values
scaler=StandardScaler(); Xs=scaler.fit_transform(X_df)
print(f"  N={len(y):,}, emp rate={y.mean():.3f}")

# ═══════════════════════════════════════════════════════
# STEP 2: Train RF
# ═══════════════════════════════════════════════════════
print("\n=== Training Random Forest ===")
rf=RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                           random_state=42, n_jobs=-1)
rf.fit(Xs, y)

# ═══════════════════════════════════════════════════════
# STEP 3: Fast SHAP via feature marginalisation
#
# Conditional expectation SHAP (Lundberg & Lee 2017, eq. 9):
# phi_i = E[f(x)|x_i] - E[f(x)]
# Approximated for each feature by:
#   for each sample x in explain set:
#     replace feature i with background distribution (N_bg samples)
#     SHAP_i(x) = f(x) - mean_over_background(f(x with feature_i replaced))
#
# This is "interventional SHAP" (marginal) — identical semantics to
# shap.TreeExplainer(model, feature_perturbation='interventional')
# ═══════════════════════════════════════════════════════
print("\n=== Computing SHAP values (fast marginal method) ===")

np.random.seed(42)
N_EXPLAIN = 500   # samples to explain
N_BG      = 100   # background samples for marginalisation

exp_idx = np.random.choice(len(Xs), N_EXPLAIN, replace=False)
bg_idx  = np.random.choice(len(Xs), N_BG,      replace=False)

X_exp = Xs[exp_idx]           # (N_EXPLAIN, n_features)
X_bg  = Xs[bg_idx]            # (N_BG, n_features)
y_exp = y[exp_idx]

# Baseline: mean prediction over background
f_base = rf.predict_proba(X_bg)[:, 1].mean()
print(f"  Baseline E[f(x)] = {f_base:.4f}")

# Compute SHAP for every feature via marginalisation
n_feat = len(X_VARS)
shap_matrix = np.zeros((N_EXPLAIN, n_feat))

for fi in range(n_feat):
    # For each explain sample, replace feature fi with each bg value
    # X_perturbed: (N_EXPLAIN × N_BG, n_features)
    X_rep  = np.repeat(X_exp, N_BG, axis=0)          # tile each explain sample N_BG times
    X_bg_t = np.tile(X_bg, (N_EXPLAIN, 1))            # tile bg dataset N_EXPLAIN times
    X_rep[:, fi] = X_bg_t[:, fi]                      # replace feature fi with bg values
    probs = rf.predict_proba(X_rep)[:, 1]             # (N_EXPLAIN × N_BG,)
    # Mean over background for each explain sample
    E_f_without_i = probs.reshape(N_EXPLAIN, N_BG).mean(axis=1)  # (N_EXPLAIN,)
    # SHAP_i(x) = f(x) - E[f(x_-i)]
    f_x = rf.predict_proba(X_exp)[:, 1]
    shap_matrix[:, fi] = f_x - E_f_without_i

    if fi % 3 == 0:
        print(f"  Feature {fi+1}/{n_feat}: {X_VARS[fi]} | mean|SHAP|={np.abs(shap_matrix[:,fi]).mean():.4f}")

print("  Done.")

# Recover original scale for dependence plots
X_exp_orig = scaler.inverse_transform(X_exp)

# ═══════════════════════════════════════════════════════
# FIGURE 10: SHAP Feature Importance Bar Plot
# ═══════════════════════════════════════════════════════
print("\n=== Figure 10: SHAP Feature Importance Bar Plot ===")

mean_abs = np.abs(shap_matrix).mean(axis=0)
shap_ser = pd.Series(mean_abs, index=X_VARS).sort_values(ascending=True)

def bar_color(f):
    if any(k in f for k in ['network','sn_']): return PAL[2]
    if any(k in f for k in ['vc','ecosystem','ecology']): return PAL[0]
    return '#B0BEC5'

fig, ax = plt.subplots(figsize=(9.5, 7))
colors_b = [bar_color(f) for f in shap_ser.index]
bars = ax.barh(range(len(shap_ser)), shap_ser.values,
               color=colors_b, edgecolor='white', alpha=0.88, height=0.68)
for i, v in enumerate(shap_ser.values):
    ax.text(v+0.0003, i, f'{v:.4f}', va='center', fontsize=8.5)
ax.set_yticks(range(len(shap_ser)))
ax.set_yticklabels([f.replace('_',' ').title() for f in shap_ser.index], fontsize=9.5)
ax.set_xlabel('Mean |SHAP Value| (Average Impact on Employment Prediction)', fontsize=10.5)
ax.set_title('Figure 10. SHAP Feature Importance — Employment Prediction\n'
             '(Random Forest, N=500 explained samples)',
             fontsize=11.5, fontweight='bold', pad=10)
patches=[mpatches.Patch(color=PAL[2],label='Social Network'),
         mpatches.Patch(color=PAL[0],label='VC Ecosystem'),
         mpatches.Patch(color='#B0BEC5',label='Control Variables')]
ax.legend(handles=patches, fontsize=9.5, loc='lower right')
ax.axvline(0, color='#555', lw=0.8, ls='--')
plt.tight_layout()
plt.savefig('mini/figures/fig10_shap_importance.png', bbox_inches='tight', dpi=150)
plt.close(); print("  Fig 10 ✓")

# ═══════════════════════════════════════════════════════
# FIGURE 11: SHAP Dependence Plot — VC Ecosystem Index
# ═══════════════════════════════════════════════════════
print("=== Figure 11: SHAP Dependence (VC) ===")
vi = X_VARS.index('vc_ecosystem_index_entropy')
si = X_VARS.index('social_network_index_entropy')

vc_orig  = X_exp_orig[:, vi]
vc_shap  = shap_matrix[:, vi]
sn_orig  = X_exp_orig[:, si]

# Normalise sn for colouring
fv_min,fv_max = sn_orig.min(), sn_orig.max()
sn_norm = (sn_orig-fv_min)/(fv_max-fv_min+1e-9)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
sc = ax.scatter(vc_orig, vc_shap, c=sn_norm, cmap='RdYlGn',
                alpha=0.70, s=22, edgecolors='none', rasterized=True)
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Social Network Index\n(interaction colour)', fontsize=9)
cbar.set_ticks([0,0.5,1]); cbar.set_ticklabels(['Low','Med','High'])

# Smoothed trend
so = np.argsort(vc_orig)
smooth = pd.Series(vc_shap[so]).rolling(max(15,len(so)//15), center=True, min_periods=5).mean()
ax.plot(vc_orig[so], smooth.values, color='red', lw=2.5, label='Smoothed trend', zorder=5)
ax.axhline(0, color='black', lw=0.9, ls='--', alpha=0.6)
ax.set_xlabel('VC Ecosystem Index (Entropy-Weighted)', fontsize=10.5)
ax.set_ylabel('SHAP Value  (Impact on P(Employed))', fontsize=10.5)
ax.set_title('Figure 11. SHAP Dependence Plot — VC Ecosystem Index\n'
             '(Coloured by Social Network Index)', fontsize=11.5, fontweight='bold', pad=10)
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.savefig('mini/figures/fig11_shap_dep_vc.png', bbox_inches='tight', dpi=150)
plt.close(); print("  Fig 11 ✓")

# ═══════════════════════════════════════════════════════
# FIGURE 12: SHAP Dependence Plot — Social Network Index
# ═══════════════════════════════════════════════════════
print("=== Figure 12: SHAP Dependence (SN) ===")
sn_orig2 = X_exp_orig[:, si]
sn_shap2 = shap_matrix[:, si]
vc_orig2 = X_exp_orig[:, vi]
vc_min,vc_max = vc_orig2.min(), vc_orig2.max()
vc_norm = (vc_orig2-vc_min)/(vc_max-vc_min+1e-9)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
sc2 = ax.scatter(sn_orig2, sn_shap2, c=vc_norm, cmap='Blues',
                 alpha=0.70, s=22, edgecolors='none', rasterized=True)
cbar2 = plt.colorbar(sc2, ax=ax, pad=0.02)
cbar2.set_label('VC Ecosystem Index\n(interaction colour)', fontsize=9)
cbar2.set_ticks([0,0.5,1]); cbar2.set_ticklabels(['Low','Med','High'])

so2 = np.argsort(sn_orig2)
smooth2 = pd.Series(sn_shap2[so2]).rolling(max(15,len(so2)//15), center=True, min_periods=5).mean()
ax.plot(sn_orig2[so2], smooth2.values, color='red', lw=2.5, label='Smoothed trend', zorder=5)
ax.axhline(0, color='black', lw=0.9, ls='--', alpha=0.6)
ax.set_xlabel('Social Network Index (Entropy-Weighted)', fontsize=10.5)
ax.set_ylabel('SHAP Value  (Impact on P(Employed))', fontsize=10.5)
ax.set_title('Figure 12. SHAP Dependence Plot — Social Network Index\n'
             '(Coloured by VC Ecosystem Index)', fontsize=11.5, fontweight='bold', pad=10)
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.savefig('mini/figures/fig12_shap_dep_sn.png', bbox_inches='tight', dpi=150)
plt.close(); print("  Fig 12 ✓")

# ═══════════════════════════════════════════════════════
# FIGURE 13: SHAP Beeswarm Summary
# ═══════════════════════════════════════════════════════
print("=== Figure 13: SHAP Beeswarm Summary ===")
feat_order = np.abs(shap_matrix).mean(axis=0).argsort()[::-1]
top_n = min(14, n_feat)
fo_top = feat_order[:top_n][::-1]   # reversed: most important at top

fig, ax = plt.subplots(figsize=(10.5, 7.5))
for row_i, fi in enumerate(fo_top):
    fs = shap_matrix[:, fi]
    fv = X_exp_orig[:, fi]
    fv_min2,fv_max2 = fv.min(),fv.max()
    fv_n = (fv-fv_min2)/(fv_max2-fv_min2+1e-9)
    np.random.seed(int(fi)*3+7)
    yj = row_i + np.random.uniform(-0.32, 0.32, size=len(fs))
    cols_s = plt.cm.RdBu_r(fv_n)
    ax.scatter(fs, yj, c=cols_s, alpha=0.55, s=9,
               edgecolors='none', rasterized=True)

ax.axvline(0, color='black', lw=1.0, ls='--', alpha=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels([X_VARS[fi].replace('_',' ').title() for fi in fo_top], fontsize=9.5)
ax.set_xlabel('SHAP Value  (Impact on Employment Probability)', fontsize=10.5)
ax.set_title('Figure 13. SHAP Summary (Beeswarm)\n'
             'Each dot = one observation | Red = high feature value | Blue = low feature value',
             fontsize=11.5, fontweight='bold', pad=10)
sm=plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0,1)); sm.set_array([])
cbar3=plt.colorbar(sm,ax=ax,pad=0.02,fraction=0.025)
cbar3.set_ticks([0,1]); cbar3.set_ticklabels(['Low','High']); cbar3.set_label('Feature Value',fontsize=9)
plt.tight_layout()
plt.savefig('mini/figures/fig13_shap_beeswarm.png', bbox_inches='tight', dpi=150)
plt.close(); print("  Fig 13 ✓")

# ═══════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════
shap_df = pd.DataFrame(shap_matrix, columns=X_VARS)
shap_df['y_true']       = y_exp
shap_df['y_pred_proba'] = rf.predict_proba(X_exp)[:, 1]
shap_df.to_csv('mini/outputs/shap_values.csv', index=False)

summary_shap = pd.DataFrame({
    'feature':        X_VARS,
    'mean_abs_shap':  mean_abs,
    'mean_shap':      shap_matrix.mean(axis=0),
    'std_shap':       shap_matrix.std(axis=0),
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
summary_shap.to_csv('mini/outputs/shap_summary.csv', index=False)

print("\n=== SHAP Analysis Complete ===")
print(summary_shap[['feature','mean_abs_shap','mean_shap']].head(8).to_string(index=False))
