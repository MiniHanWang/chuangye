"""
Venture Capital Ecosystem, Social Networks, and Employment Outcomes:
Micro Evidence from China — Full Research Pipeline (v3, FINAL)
Author Mini Han Wang
"""
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from openpyxl import load_workbook as _load_wb

warnings.filterwarnings('ignore')
os.makedirs('mini/figures', exist_ok=True)
os.makedirs('mini/outputs', exist_ok=True)

PAL = ['#2E75B6','#ED7D31','#70AD47','#5B9BD5','#FFC000','#A9D18E','#FF7575','#9DC3E6']
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':120,'font.size':10})

# ═══════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("=== STEP 1: Data Loading ===")
MISS = {'拒绝回答','不知道','未调查','入户面访未调查','不适用','电话调查未调查',
        '不接受调查','没有回答', 'None', None}

_wb = _load_wb('/mnt/user-data/uploads/就业.xlsx', read_only=True)
_ws = _wb['Sheet1']
_all = list(_ws.iter_rows(values_only=True))
_codes = [str(c) if c is not None else '' for c in _all[0]]
# De-duplicate: append __dupN for repeats
_seen = {}; _unique = []
for c in _codes:
    if c not in _seen: _seen[c] = 0; _unique.append(c)
    else: _seen[c] += 1; _unique.append(f"{c}__dup{_seen[c]}")

raw = pd.DataFrame(_all[2:], columns=_unique)

# Replace missing strings with NaN
for col in raw.columns:
    if raw[col].dtype == object:
        raw[col] = raw[col].apply(lambda x: np.nan if x in MISS or (isinstance(x,str) and x.strip() in MISS) else x)

print(f"  Raw: {raw.shape}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: DEPENDENT VARIABLE — EMPLOYMENT
# ═══════════════════════════════════════════════════════════════════
print("\n=== STEP 2: Employment Variables ===")

# a8a (col 8): annual income — some entries are text "收入高于百万位数"
raw['income_annual'] = pd.to_numeric(raw['a8a'], errors='coerce')
raw.loc[raw['income_annual'] < 0, 'income_annual'] = np.nan
# Treat "收入高于百万位数" as >1,000,000
raw.loc[(raw['a8a'].astype(str).str.contains('百万', na=False)) & raw['income_annual'].isna(),
        'income_annual'] = 1_500_000

def classify_emp(row):
    inc = row['income_annual']
    a53 = str(row.get('a53',''))
    a58 = str(row.get('a58',''))
    # In-person respondents: a53 gives direct last-week work status
    if '从事了以取得经济收入为目的的工作' in a53:
        return 1
    if '停薪休假' in a53 or '带薪休假' in a53:
        # temporarily absent — still employed if has income
        return 1 if (pd.notna(inc) and inc > 0) else 0
    if '未从事任何以获得经济收入为目的的工作' in a53:
        # did not work last week; check a58 and income
        if '目前从事非农工作' in a58 or '目前务农' in a58:
            return 1  # currently working, just not last week
        if pd.notna(inc) and inc > 0:
            return 1
        return 0
    # Phone respondents: use income as primary indicator
    if pd.notna(inc) and inc > 0:
        return 1
    if pd.notna(inc) and inc == 0:
        # Check a58 for currently working
        if '目前从事非农工作' in a58 or '目前务农' in a58:
            return 1
        if '没有工作' in a58 or '从未工作' in a58:
            return 0
        return 0
    # Hours worked
    hrs = pd.to_numeric(row.get('a54'), errors='coerce')
    if pd.notna(hrs) and hrs > 0:
        return 1
    return np.nan

raw['employment_binary'] = raw.apply(classify_emp, axis=1)

# Log income (only positive)
raw['log_income'] = np.where(raw['income_annual'] > 0, np.log(raw['income_annual']), np.nan)

# Non-farm employment from isco
raw['isco_str'] = raw['isco08a59d'].astype(str).fillna('')
NONFARM_KW = ['经理','专业','技术','文职','销售','服务','操作员','工人','驾驶','办事','工程','医','教','律']
raw['employment_nonfarm'] = raw['isco_str'].apply(
    lambda x: 1 if any(k in x for k in NONFARM_KW) else (0 if x not in ['','nan'] else np.nan))

ec = raw['employment_binary'].value_counts()
print(f"  Employment: {ec.to_dict()}, missing={raw['employment_binary'].isna().sum()}")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: SOCIAL NETWORK INDEX
# ═══════════════════════════════════════════════════════════════════
print("\n=== STEP 3: Social Network Index ===")

# Columns 80-83 are ALREADY recoded numeric SN dimensions
# col80=a31a(recoded), col81=a31b(recoded), col82=a33(recoded), col83=a313(recoded)
raw['sn_neighbor'] = pd.to_numeric(raw['a31a__dup1'], errors='coerce')   # col80
raw['sn_friend']   = pd.to_numeric(raw['a31b__dup1'], errors='coerce')   # col81
raw['sn_trust']    = pd.to_numeric(raw['a33__dup1'],  errors='coerce')   # col82
raw['sn_learning'] = pd.to_numeric(raw['a313'],        errors='coerce')  # col83

SN_DIMS = ['sn_neighbor','sn_friend','sn_trust','sn_learning']

def entropy_idx(df_in):
    d = df_in.astype(float).copy()
    for c in d.columns:
        mn, mx = d[c].min(), d[c].max()
        d[c] = (d[c] - mn) / (mx - mn + 1e-9)
    d = d.fillna(d.median())
    n = len(d)
    p = d.div(d.sum(axis=0) + 1e-9)
    e = -(p * np.log(p + 1e-12)).sum(axis=0) / np.log(n + 1e-9)
    w = (1 - e) / ((1 - e).sum() + 1e-9)
    return d.dot(w)

def std_idx(df_in):
    d = df_in.astype(float).fillna(df_in.median())
    return pd.Series(StandardScaler().fit_transform(d).mean(axis=1), index=d.index)

def pca_idx(df_in):
    d = df_in.astype(float).fillna(df_in.median())
    sc = StandardScaler(); pca = PCA(n_components=1)
    return pd.Series(pca.fit_transform(sc.fit_transform(d)).flatten(), index=d.index)

vsn = raw[SN_DIMS].dropna()
raw['social_network_index_entropy'] = np.nan
raw['social_network_index_std']     = np.nan
raw['social_network_index_pca']     = np.nan
raw.loc[vsn.index, 'social_network_index_entropy'] = entropy_idx(vsn).values
raw.loc[vsn.index, 'social_network_index_std']     = std_idx(vsn).values
raw.loc[vsn.index, 'social_network_index_pca']     = pca_idx(vsn).values
print(f"  SN entropy: mean={raw['social_network_index_entropy'].mean():.3f}, "
      f"sd={raw['social_network_index_entropy'].std():.3f}, valid={vsn.shape[0]:,}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: VC ECOSYSTEM INDEX
# ═══════════════════════════════════════════════════════════════════
print("\n=== STEP 4: VC Ecosystem Index ===")
vc_dims = {
    'angel':        ['早期案例数（起）','早期投资金额（亿元）'],
    'vc':           ['VC 案例数（起）','VC 投资金额（亿元）'],
    'pe':           ['PE 案例数（起）','PE 投资金额（亿元）'],
    'financial':    ['金融业市场化'],
    'hr':           ['人力资源供应条件'],
    'tech':         ['技术成果市场化','知识产权保护'],
    'intermediary': ['市场中介组织的发育','维护市场的法治环境'],
}
for nm, cols in vc_dims.items():
    c = [x for x in cols if x in raw.columns]
    if c:
        raw[f'{nm}_ecology'] = raw[c].apply(pd.to_numeric, errors='coerce').mean(axis=1)

VC_DIMS = [f'{n}_ecology' for n in vc_dims if f'{n}_ecology' in raw.columns]
vvc = raw[VC_DIMS].dropna()
raw['vc_ecosystem_index_entropy'] = np.nan
raw['vc_ecosystem_index_std']     = np.nan
raw['vc_ecosystem_index_pca']     = np.nan
raw.loc[vvc.index, 'vc_ecosystem_index_entropy'] = entropy_idx(vvc).values
raw.loc[vvc.index, 'vc_ecosystem_index_std']     = std_idx(vvc).values
raw.loc[vvc.index, 'vc_ecosystem_index_pca']     = pca_idx(vvc).values
print(f"  VC entropy: mean={raw['vc_ecosystem_index_entropy'].mean():.3f}, "
      f"sd={raw['vc_ecosystem_index_entropy'].std():.3f}, valid={vvc.shape[0]:,}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: CONTROL VARIABLES
# ═══════════════════════════════════════════════════════════════════
print("\n=== STEP 5: Control Variables ===")
raw['age']    = 2023 - pd.to_numeric(raw['a3a'], errors='coerce')
raw['age_sq'] = raw['age'] ** 2
raw['gender'] = raw['a2'].map({'男':1,'女':0})

edu_map = {'没有受过任何教育':1,'小学':2,'初中':3,'高中':4,'普通高中':4,'中专':5,
           '大专':6,'大学专科（正规高等教育）':6,'大学专科（成人高等教育）':6,
           '大学本科（正规高等教育）':7,'大学本科（成人高等教育）':7,
           '研究生及以上':8,'研究生':8}
raw['education']  = raw['a7a'].map(edu_map)
raw['father_edu'] = raw['a89b'].map(edu_map)
raw['mother_edu'] = raw['a90b'].map(edu_map)

raw['health'] = raw['a15'].map({'非常不健康':1,'比较不健康':2,'一般':3,'比较健康':4,'非常健康':5})
raw['hukou_urban'] = raw['a18'].apply(
    lambda x: 1 if pd.notna(x) and '非农' in str(x) else (0 if pd.notna(x) else np.nan))
raw['married'] = raw['a69'].map({'初婚有配偶':1,'再婚有配偶':1,'未婚':0,'离婚':0,'丧偶':0,'同居':1})

# a65 (col39) = numeric economic level 0-3
raw['family_econ'] = pd.to_numeric(raw['a65'], errors='coerce')

# Car ownership
raw['has_car']   = raw['a671'].map({'是':1,'否':0,'有':1,'没有':0})
raw['has_house'] = raw['a12a'].map({'是':1,'否':0})

# Subjective class (a43a-e, cols 24-28: first occurrence, text scores)
for c in ['a43a','a43b','a43c','a43d','a43e']:
    raw[f'{c}_n'] = raw[c].apply(
        lambda x: pd.to_numeric(str(x).replace('分','').replace('层','').strip(), errors='coerce')
        if pd.notna(x) else np.nan)
raw['subjective_class'] = raw[['a43a_n','a43b_n','a43c_n','a43d_n','a43e_n']].mean(axis=1)

CONTROLS = ['age','gender','education','health','hukou_urban','married',
            'father_edu','mother_edu','family_econ','has_house','has_car','subjective_class']
CONTROLS = [c for c in CONTROLS if c in raw.columns]

# Check missing %
print("  Control missing %:")
for c in CONTROLS:
    print(f"    {c}: {raw[c].isna().mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# BUILD ANALYTIC SAMPLE
# ═══════════════════════════════════════════════════════════════════
print("\n=== Building Analytic Sample ===")
df = raw.copy()

# Median-impute controls (documented in methods)
imp = SimpleImputer(strategy='median')
df[CONTROLS] = imp.fit_transform(df[CONTROLS])

# Require valid outcome and both main exposures
KEY = ['employment_binary','social_network_index_entropy','vc_ecosystem_index_entropy']
analytic = df.dropna(subset=KEY).copy()

# Verify employment variance
emp_vals = analytic['employment_binary'].value_counts()
print(f"  Analytic N={len(analytic):,}, Employment: {emp_vals.to_dict()}")

# If employment_binary has no variance (all 1s), re-check coding
if len(emp_vals) < 2:
    print("  WARNING: No variance in employment — refining coding")
    # Only those with income>0 or clearly working = employed; income==0 AND 未从事 = unemployed
    def strict_emp(row):
        inc = row['income_annual']
        a53 = str(row.get('a53',''))
        a58 = str(row.get('a58',''))
        if '目前没有工作' in a58 or '从未工作' in a58:
            return 0
        if '未从事' in a53 and (pd.isna(inc) or inc == 0):
            return 0
        if pd.notna(inc) and inc > 0:
            return 1
        if '从事了' in a53 or '目前从事' in a58:
            return 1
        return 0   # default to 0 if truly ambiguous
    df['employment_binary'] = df.apply(strict_emp, axis=1)
    analytic = df.dropna(subset=['social_network_index_entropy','vc_ecosystem_index_entropy']).copy()
    print(f"  After re-code: {analytic['employment_binary'].value_counts().to_dict()}")

emp_rate = analytic['employment_binary'].mean()
print(f"  Employment rate: {emp_rate:.3f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: ECONOMETRIC MODELS
# ═══════════════════════════════════════════════════════════════════
print("\n=== STEP 6: Econometric Models ===")
X_VARS = ['social_network_index_entropy','vc_ecosystem_index_entropy'] + CONTROLS

def prep_xy(data, y_col):
    sub = data.dropna(subset=[y_col]).copy()
    X = sub[X_VARS].copy()
    X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=X_VARS, index=sub.index)
    y = sub[y_col].values
    Xs = StandardScaler().fit_transform(X)
    return Xs, y

# Model 1: Logit
print("  Model 1: Logistic Regression (Employment Binary)")
Xs1, y1 = prep_xy(analytic, 'employment_binary')
lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
lr.fit(Xs1, y1)
kf = StratifiedKFold(5, shuffle=True, random_state=42)
lr_auc = cross_val_score(lr, Xs1, y1, cv=kf, scoring='roc_auc').mean()
lr_acc = cross_val_score(lr, Xs1, y1, cv=kf, scoring='accuracy').mean()
lr_coefs = pd.DataFrame({'variable':X_VARS, 'coef':lr.coef_[0],
                          'odds_ratio':np.exp(lr.coef_[0])}).round(4)
print(f"    AUC={lr_auc:.3f}, Acc={lr_acc:.3f}, N={len(y1):,}")

# Model 2: OLS (log income)
print("  Model 2: OLS (Log Income)")
inc_analytic = analytic[analytic['log_income'].notna() & (analytic['log_income'] > 0)]
Xs2, y2 = prep_xy(inc_analytic, 'log_income')
ols = LinearRegression()
ols.fit(Xs2, y2)
ols_r2 = cross_val_score(ols, Xs2, y2, cv=5, scoring='r2').mean()
ols_coefs = pd.DataFrame({'variable':X_VARS,'coef':ols.coef_}).round(4)
print(f"    R²={ols_r2:.3f}, N={len(y2):,}")

# Model 3: Bootstrap Mediation
print("  Model 3: Bootstrap Mediation (N=500)")
med_data = analytic.copy()
med_data[X_VARS] = SimpleImputer(strategy='median').fit_transform(med_data[X_VARS])

def get_med(d):
    X  = StandardScaler().fit_transform(d[['vc_ecosystem_index_entropy']+CONTROLS].values)
    M  = d['social_network_index_entropy'].values
    Y  = d['employment_binary'].values
    a  = LinearRegression().fit(X, M).coef_[0]
    XM = np.column_stack([X, M])
    bc = LogisticRegression(max_iter=500, C=1.0).fit(XM, Y)
    b  = bc.coef_[0][-1]
    cp = bc.coef_[0][0]
    return a, b, cp, a*b

a0,b0,cp0,ind0 = get_med(med_data)
boot = []
for _ in range(500):
    try: boot.append(get_med(resample(med_data, random_state=None))[3])
    except: pass
boot = np.array(boot)
ci_lo, ci_hi = np.percentile(boot,[2.5,97.5])
med_sig = not (ci_lo < 0 < ci_hi)
print(f"    Indirect={ind0:.4f} 95%CI [{ci_lo:.4f},{ci_hi:.4f}] sig={med_sig}")

# ═══════════════════════════════════════════════════════════════════
# STEP 7: MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════
print("\n=== STEP 7: Machine Learning ===")
ml_models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0),
    'Random Forest':       RandomForestClassifier(200, max_depth=8, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
}
ml_results = {}
for mn, mod in ml_models.items():
    auc = cross_val_score(mod, Xs1, y1, cv=kf, scoring='roc_auc').mean()
    acc = cross_val_score(mod, Xs1, y1, cv=kf, scoring='accuracy').mean()
    f1  = cross_val_score(mod, Xs1, y1, cv=kf, scoring='f1_weighted').mean()
    ml_results[mn] = {'AUC':round(auc,4),'Accuracy':round(acc,4),'F1':round(f1,4)}
    print(f"  {mn}: AUC={auc:.3f} Acc={acc:.3f} F1={f1:.3f}")

rf_fit = RandomForestClassifier(200, max_depth=8, random_state=42, n_jobs=-1)
rf_fit.fit(Xs1, y1)
feat_imp = pd.Series(rf_fit.feature_importances_, index=X_VARS).sort_values(ascending=False)
print(f"  Top feature: {feat_imp.index[0]} ({feat_imp.iloc[0]:.4f})")

# ═══════════════════════════════════════════════════════════════════
# STEP 8: FIGURES
# ═══════════════════════════════════════════════════════════════════
print("\n=== STEP 8: Generating Figures ===")
FP = 'mini/figures/'

# ── Fig 1: Conceptual Framework ──
fig, ax = plt.subplots(figsize=(12, 5.8))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
bx = [(0.03,0.50,0.24,0.32,PAL[0],'VC Ecosystem\nIndex\n(Province Level)'),
      (0.38,0.50,0.24,0.32,PAL[2],'Social Network\nIndex\n(Individual Level)'),
      (0.73,0.50,0.24,0.32,PAL[1],'Employment\nOutcomes\n(Binary / Log Income)'),
      (0.30,0.06,0.40,0.30,'#E8F5E9','Controls: Age, Gender,\nEducation, Health, Hukou\nMarital Status, Class')]
for x,y,w,h,c,lbl in bx:
    ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025',
                 lw=1.8,edgecolor='#555',facecolor=c,alpha=0.92))
    ax.text(x+w/2,y+h/2,lbl,ha='center',va='center',fontsize=9.5,fontweight='bold',color='#111')
kw_arr = dict(arrowstyle='->', lw=2.2)
ax.annotate('',xy=(0.38,0.66),xytext=(0.27,0.66),arrowprops=dict(**kw_arr,color='#333'))
ax.annotate('',xy=(0.73,0.66),xytext=(0.62,0.66),arrowprops=dict(**kw_arr,color='#333'))
ax.annotate('',xy=(0.85,0.50),xytext=(0.15,0.50),
            arrowprops=dict(arrowstyle='->',color=PAL[3],lw=1.8,linestyle='dashed'))
ax.annotate('',xy=(0.50,0.50),xytext=(0.50,0.36),arrowprops=dict(arrowstyle='->',color='#888',lw=1.4))
ax.text(0.325,0.695,'Path a',ha='center',fontsize=8.5,fontstyle='italic')
ax.text(0.675,0.695,'Path b',ha='center',fontsize=8.5,fontstyle='italic')
ax.text(0.500,0.445,"Path c' (direct)",ha='center',fontsize=8,color=PAL[3],fontstyle='italic')
ax.set_title('Figure 1. Conceptual Framework: VC Ecosystem, Social Networks, and Employment Outcomes',
             fontsize=11.5,fontweight='bold',pad=10)
plt.tight_layout(); plt.savefig(FP+'fig1_framework.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 1 ✓")

# ── Fig 2: SN Distribution ──
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
for ax,(col,sub) in zip(axes,[('social_network_index_entropy','(a) Entropy-Weighted Index'),
                               ('social_network_index_pca','(b) PCA Index')]):
    d = analytic[col].dropna()
    ax.hist(d,bins=45,color=PAL[2],edgecolor='white',alpha=0.85,lw=0.4)
    ax.axvline(d.mean(),color='red',ls='--',lw=1.8,label=f'Mean={d.mean():.3f}')
    ax.axvline(d.median(),color='orange',ls=':',lw=1.8,label=f'Median={d.median():.3f}')
    ax.set_xlabel('Index Value',fontsize=10); ax.set_ylabel('Frequency',fontsize=10)
    ax.set_title(sub,fontsize=10.5,fontweight='bold'); ax.legend(fontsize=9)
fig.suptitle('Figure 2. Distribution of Social Network Index',fontsize=12,fontweight='bold',y=1.02)
plt.tight_layout(); plt.savefig(FP+'fig2_sn_dist.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 2 ✓")

# ── Fig 3: VC Distribution ──
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
for ax,(col,sub) in zip(axes,[('vc_ecosystem_index_entropy','(a) Entropy-Weighted Index'),
                               ('vc_ecosystem_index_pca','(b) PCA Index')]):
    d = analytic[col].dropna()
    ax.hist(d,bins=35,color=PAL[0],edgecolor='white',alpha=0.85,lw=0.4)
    ax.axvline(d.mean(),color='red',ls='--',lw=1.8,label=f'Mean={d.mean():.3f}')
    ax.axvline(d.median(),color='orange',ls=':',lw=1.8,label=f'Median={d.median():.3f}')
    ax.set_xlabel('Index Value',fontsize=10); ax.set_ylabel('Frequency',fontsize=10)
    ax.set_title(sub,fontsize=10.5,fontweight='bold'); ax.legend(fontsize=9)
fig.suptitle('Figure 3. Distribution of VC Ecosystem Index',fontsize=12,fontweight='bold',y=1.02)
plt.tight_layout(); plt.savefig(FP+'fig3_vc_dist.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 3 ✓")

# ── Fig 4: Employment Rate by VC Quartile ──
fig, ax = plt.subplots(figsize=(8.5,5.5))
a4 = analytic.copy()
a4['vc_q'] = pd.qcut(a4['vc_ecosystem_index_entropy'],4,
                      labels=['Q1\n(Lowest)','Q2','Q3','Q4\n(Highest)'])
rates  = a4.groupby('vc_q')['employment_binary'].mean()*100
counts = a4.groupby('vc_q')['employment_binary'].count()
bars = ax.bar(rates.index,rates.values,color=PAL[:4],edgecolor='white',width=0.62,alpha=0.9)
for bar,v,n in zip(bars,rates.values,counts.values):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
            f'{v:.1f}%\n(n={n:,})',ha='center',va='bottom',fontsize=9.5,fontweight='bold')
ax.set_xlabel('VC Ecosystem Index Quartile',fontsize=11)
ax.set_ylabel('Employment Rate (%)',fontsize=11)
ax.set_ylim(0,max(rates.values)*1.22)
ax.set_title('Figure 4. Employment Rate by VC Ecosystem Level',fontsize=12,fontweight='bold')
plt.tight_layout(); plt.savefig(FP+'fig4_emp_by_vc.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 4 ✓")

# ── Fig 5: Coefficient Plot ──
fig, axes = plt.subplots(1,2,figsize=(14,6.5))
for ax,(coefs,title,vc) in zip(axes,[
    (lr_coefs,'(a) Logit: Employment (Odds Ratios)',True),
    (ols_coefs,'(b) OLS: Log Income (Std. Coefficients)',False)]):
    vc_col = 'odds_ratio' if vc else 'coef'; ref=1 if vc else 0
    top = coefs.sort_values(vc_col,ascending=True).tail(12)
    labs = [v.replace('_',' ').title() for v in top['variable']]
    vals = top[vc_col].values - ref
    cols_b = [PAL[1] if v>0 else '#90CAF9' for v in vals]
    ax.barh(range(len(labs)),vals,color=cols_b,edgecolor='white',alpha=0.85,height=0.65)
    ax.axvline(0,color='#333',lw=0.9,ls='--')
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=8.5)
    ax.set_xlabel('Effect (vs. reference)',fontsize=9.5)
    ax.set_title(title,fontsize=10.5,fontweight='bold')
fig.suptitle('Figure 5. Regression Coefficient Plots',fontsize=12,fontweight='bold',y=1.01)
plt.tight_layout(); plt.savefig(FP+'fig5_coefs.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 5 ✓")

# ── Fig 6: Mediation Diagram ──
fig, ax = plt.subplots(figsize=(10.5,5.5))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
for x,y,w,h,c,lbl in [(0.03,0.44,0.26,0.28,PAL[0],'VC Ecosystem\nIndex'),
                       (0.37,0.44,0.26,0.28,PAL[2],'Social Network\nIndex'),
                       (0.71,0.44,0.26,0.28,PAL[1],'Employment\nBinary')]:
    ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025',
                 lw=1.8,edgecolor='#555',facecolor=c,alpha=0.9))
    ax.text(x+w/2,y+h/2,lbl,ha='center',va='center',fontsize=11,fontweight='bold')
ax.annotate('',xy=(0.37,0.58),xytext=(0.29,0.58),arrowprops=dict(arrowstyle='->',color='#222',lw=2.2))
ax.annotate('',xy=(0.71,0.58),xytext=(0.63,0.58),arrowprops=dict(arrowstyle='->',color='#222',lw=2.2))
ax.annotate('',xy=(0.85,0.44),xytext=(0.15,0.44),
            arrowprops=dict(arrowstyle='->',color=PAL[3],lw=2,linestyle='dashed'))
ax.text(0.33,0.655,f'a = {a0:.4f}',ha='center',fontsize=10,fontweight='bold')
ax.text(0.67,0.655,f'b = {b0:.4f}',ha='center',fontsize=10,fontweight='bold')
ax.text(0.50,0.365,f"c' = {cp0:.4f}",ha='center',fontsize=10,color=PAL[3],fontweight='bold')
sig_lbl='✓ Significant' if med_sig else '✗ Not Significant'
ci_clr='#1B5E20' if med_sig else '#B71C1C'
ax.text(0.50,0.17,
        f'Indirect Effect (a×b) = {ind0:.4f}\n95% Bootstrap CI [{ci_lo:.4f}, {ci_hi:.4f}]   {sig_lbl}',
        ha='center',fontsize=10.5,color=ci_clr,
        bbox=dict(boxstyle='round,pad=0.4',facecolor='#FFFDE7',edgecolor='#BDBDBD',alpha=0.95))
ax.set_title('Figure 6. Bootstrap Mediation: VC Ecosystem → Social Network → Employment',
             fontsize=11.5,fontweight='bold',pad=10)
plt.tight_layout(); plt.savefig(FP+'fig6_mediation.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 6 ✓")

# ── Fig 7: Feature Importance ──
fig, ax = plt.subplots(figsize=(9.5,6.5))
t12 = feat_imp.head(12)
labs = [f.replace('_',' ').title() for f in t12.index]
bc = [PAL[2] if any(k in f for k in ['network','sn_','neighbor','friend','trust','learning'])
      else (PAL[0] if any(k in f for k in ['vc','ecosystem','ecology']) else '#B0BEC5')
      for f in t12.index]
ax.barh(range(len(labs)),t12.values,color=bc,edgecolor='white',alpha=0.88,height=0.68)
for i,v in enumerate(t12.values):
    ax.text(v+0.0005,i,f'{v:.4f}',va='center',fontsize=8.5)
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=9.5)
ax.set_xlabel('Mean Decrease in Gini Impurity',fontsize=10)
ax.set_title('Figure 7. Random Forest Feature Importance (Top 12)',fontsize=12,fontweight='bold')
patches=[mpatches.Patch(color=PAL[2],label='Social Network'),
         mpatches.Patch(color=PAL[0],label='VC Ecosystem'),
         mpatches.Patch(color='#B0BEC5',label='Controls')]
ax.legend(handles=patches,fontsize=9.5,loc='lower right')
plt.tight_layout(); plt.savefig(FP+'fig7_importance.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 7 ✓")

# ── Fig 8: Income vs SN ──
fig, ax = plt.subplots(figsize=(8.5,5.5))
p8 = analytic[analytic['log_income'].notna() & (analytic['log_income']>0)].dropna(
    subset=['social_network_index_entropy'])
if len(p8) > 20:
    ax.scatter(p8['social_network_index_entropy'],p8['log_income'],
               alpha=0.18,color=PAL[2],s=12,rasterized=True)
    z=np.polyfit(p8['social_network_index_entropy'],p8['log_income'],1)
    xs=np.linspace(p8['social_network_index_entropy'].min(),p8['social_network_index_entropy'].max(),200)
    ax.plot(xs,np.poly1d(z)(xs),color='red',lw=2.2,label=f'Trend (β={z[0]:.3f})')
    ax.legend(fontsize=10)
ax.set_xlabel('Social Network Index (Entropy-Weighted)',fontsize=11)
ax.set_ylabel('Log Annual Income (CNY)',fontsize=11)
ax.set_title('Figure 8. Log Income vs. Social Network Index',fontsize=12,fontweight='bold')
plt.tight_layout(); plt.savefig(FP+'fig8_income_sn.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 8 ✓")

# ── Fig 9: Income vs VC ──
fig, ax = plt.subplots(figsize=(8.5,5.5))
p9 = analytic[analytic['log_income'].notna() & (analytic['log_income']>0)].dropna(
    subset=['vc_ecosystem_index_entropy'])
if len(p9) > 20:
    ax.scatter(p9['vc_ecosystem_index_entropy'],p9['log_income'],
               alpha=0.18,color=PAL[0],s=12,rasterized=True)
    z=np.polyfit(p9['vc_ecosystem_index_entropy'],p9['log_income'],1)
    xs=np.linspace(p9['vc_ecosystem_index_entropy'].min(),p9['vc_ecosystem_index_entropy'].max(),200)
    ax.plot(xs,np.poly1d(z)(xs),color='red',lw=2.2,label=f'Trend (β={z[0]:.3f})')
    ax.legend(fontsize=10)
ax.set_xlabel('VC Ecosystem Index (Entropy-Weighted)',fontsize=11)
ax.set_ylabel('Log Annual Income (CNY)',fontsize=11)
ax.set_title('Figure 9. Log Income vs. VC Ecosystem Index',fontsize=12,fontweight='bold')
plt.tight_layout(); plt.savefig(FP+'fig9_income_vc.png',bbox_inches='tight',dpi=150); plt.close()
print("  Fig 9 ✓")

# ═══════════════════════════════════════════════════════════════════
# STEP 9: TABLES
# ═══════════════════════════════════════════════════════════════════
print("\n=== STEP 9: Tables ===")

# Table 1: Variable definitions
table1 = pd.DataFrame([
    ['employment_binary','Dependent','Binary: 1=employed, 0=unemployed (income>0, or working last week)','a8a, a53, a58, a62'],
    ['log_income','Dependent','Natural log of annual personal income (CNY, 2022)','a8a'],
    ['social_network_index_entropy','Main IV','Composite SN index via entropy weighting (4 dimensions)','a31a, a31b, a33, a313 (recoded cols 80-83)'],
    ['vc_ecosystem_index_entropy','Main IV','Composite VC ecosystem index via entropy weighting (7 dimensions)','Province VC + marketization data'],
    ['sn_neighbor','SN Component','Neighbor interaction frequency (1=never to 7=daily)','a31a'],
    ['sn_friend','SN Component','Friend interaction frequency (1=never to 7=daily)','a31b'],
    ['sn_trust','SN Component','Generalised social trust (1=strongly disagree to 5=strongly agree)','a33'],
    ['sn_learning','SN Component','Social learning frequency (1=never to 5=very frequent)','a313'],
    ['angel_ecology','VC Component','Angel/early-stage ecology (avg. deal count + volume)','早期案例数, 早期投资金额'],
    ['vc_ecology','VC Component','VC-round ecology (avg. deal count + volume)','VC案例数, VC投资金额'],
    ['pe_ecology','VC Component','PE-round ecology (avg. deal count + volume)','PE案例数, PE投资金额'],
    ['intermediary_ecology','VC Component','Market intermediary & legal environment','市场中介组织发育, 法治环境'],
    ['tech_ecology','VC Component','Technology commercialisation & IP protection','技术成果市场化, 知识产权保护'],
    ['age','Control','Respondent age in years','a3a'],
    ['gender','Control','Male=1, Female=0','a2'],
    ['education','Control','Education level (1=none to 8=postgraduate)','a7a'],
    ['health','Control','Self-rated health (1=very poor to 5=very good)','a15'],
    ['hukou_urban','Control','Urban hukou=1, rural=0','a18'],
    ['married','Control','Currently married=1, other=0','a69'],
    ['father_edu','Control','Father education level (1–8)','a89b'],
    ['mother_edu','Control','Mother education level (1–8)','a90b'],
    ['family_econ','Control','Household economic status relative to local average (numeric)','a65'],
    ['has_house','Control','Owns housing (1=yes)','a12a'],
    ['has_car','Control','Owns car (1=yes)','a671'],
    ['subjective_class','Control','Mean subjective social class (1–10, avg. of a43a–e)','a43a–a43e'],
], columns=['Variable','Type','Description','Source Variable(s)'])

# Table 2: Descriptive statistics
desc_v = ['employment_binary','log_income','social_network_index_entropy','vc_ecosystem_index_entropy',
          'sn_neighbor','sn_friend','sn_trust','sn_learning',
          'age','gender','education','health','hukou_urban','married','family_econ','subjective_class']
desc_v = [v for v in desc_v if v in analytic.columns]
table2 = analytic[desc_v].describe().T.round(3)
table2.columns = ['N','Mean','Std Dev','Min','P25','Median','P75','Max']
table2['N'] = table2['N'].astype(int)

# Table 3: Correlation matrix
corr_v = ['employment_binary','log_income','social_network_index_entropy','vc_ecosystem_index_entropy',
          'age','education','health','hukou_urban','family_econ']
corr_v = [v for v in corr_v if v in analytic.columns]
table3 = analytic[corr_v].corr().round(3)

# Table 4 & 5: Regression
table4 = lr_coefs[['variable','coef','odds_ratio']].copy()
table4.columns = ['Variable','Log-Odds','Odds Ratio']
table4['Variable'] = table4['Variable'].str.replace('_',' ').str.title()

table5 = ols_coefs.copy()
table5.columns = ['Variable','Std. Coefficient']
table5['Variable'] = table5['Variable'].str.replace('_',' ').str.title()

# Table 6: Mediation
table6 = pd.DataFrame([
    ['Path a: VC Ecosystem → Social Network', f'{a0:.4f}','—','OLS'],
    ['Path b: Social Network → Employment (ctrl. VC)', f'{b0:.4f}','—','Logit'],
    ["Path c': Direct VC → Employment (ctrl. SN)", f'{cp0:.4f}','—','Logit'],
    ['Indirect Effect (a × b)', f'{ind0:.4f}', f'[{ci_lo:.4f}, {ci_hi:.4f}]','Bootstrap (N=500)'],
    ['Mediation Significant?', str(med_sig),'—','95% CI excludes 0'],
], columns=['Effect','Estimate','95% CI','Method'])

# Table 7: ML performance
table7 = pd.DataFrame(
    [(k, v['AUC'], v['Accuracy'], v['F1']) for k,v in ml_results.items()],
    columns=['Model','5-Fold CV AUC','5-Fold CV Accuracy','5-Fold CV F1'])

with pd.ExcelWriter('mini/outputs/regression_results.xlsx',engine='openpyxl') as wr:
    table1.to_excel(wr, sheet_name='Table1_Variables',   index=False)
    table2.to_excel(wr, sheet_name='Table2_Descriptive')
    table3.to_excel(wr, sheet_name='Table3_Correlation')
    table4.to_excel(wr, sheet_name='Table4_Logit',       index=False)
    table5.to_excel(wr, sheet_name='Table5_OLS',         index=False)
    table6.to_excel(wr, sheet_name='Table6_Mediation',   index=False)
    table7.to_excel(wr, sheet_name='Table7_ML',          index=False)
    feat_imp.to_frame('RF_Importance').to_excel(wr, sheet_name='FeatureImportance')

raw.to_csv('mini/outputs/cleaned_data.csv', index=False)
table1.to_excel('mini/outputs/variable_codebook.xlsx', index=False)

# Save summary for report
SN_COEF = float(lr_coefs[lr_coefs.variable=='social_network_index_entropy']['coef'].values[0])
SN_OR   = float(lr_coefs[lr_coefs.variable=='social_network_index_entropy']['odds_ratio'].values[0])
VC_COEF = float(lr_coefs[lr_coefs.variable=='vc_ecosystem_index_entropy']['coef'].values[0])
VC_OR   = float(lr_coefs[lr_coefs.variable=='vc_ecosystem_index_entropy']['odds_ratio'].values[0])
SN_OLS  = float(ols_coefs[ols_coefs.variable=='social_network_index_entropy']['coef'].values[0])
VC_OLS  = float(ols_coefs[ols_coefs.variable=='vc_ecosystem_index_entropy']['coef'].values[0])
EMP_RATE= float(emp_rate)
MEAN_SN = float(analytic['social_network_index_entropy'].mean())
SD_SN   = float(analytic['social_network_index_entropy'].std())
MEAN_VC = float(analytic['vc_ecosystem_index_entropy'].mean())
SD_VC   = float(analytic['vc_ecosystem_index_entropy'].std())

summary = dict(
    n_raw=int(len(raw)), n_analytic=int(len(analytic)), emp_rate=EMP_RATE,
    mean_sn=MEAN_SN, sd_sn=SD_SN, mean_vc=MEAN_VC, sd_vc=SD_VC,
    sn_logit_coef=SN_COEF, sn_logit_or=SN_OR,
    vc_logit_coef=VC_COEF, vc_logit_or=VC_OR,
    sn_ols_coef=SN_OLS, vc_ols_coef=VC_OLS,
    logit_auc=float(lr_auc), logit_acc=float(lr_acc),
    ols_r2=float(ols_r2), n_income=int(len(y2)),
    mediation_a=float(a0), mediation_b=float(b0), mediation_cp=float(cp0),
    mediation_indirect=float(ind0), mediation_ci_lo=float(ci_lo),
    mediation_ci_hi=float(ci_hi), mediation_sig=bool(med_sig),
    rf_auc=float(ml_results['Random Forest']['AUC']),
    gb_auc=float(ml_results['Gradient Boosting']['AUC']),
    ml_results=ml_results,
    top5_features=feat_imp.head(5).to_dict(),
    table2_stats={v: {'mean':float(analytic[v].mean()),'sd':float(analytic[v].std())}
                  for v in desc_v if v in analytic.columns}
)
with open('mini/outputs/results_summary.json','w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n=== All Analysis Complete ===")
print(f"  N={summary['n_analytic']:,}, Emp={summary['emp_rate']:.3f}")
print(f"  Logit AUC={summary['logit_auc']:.3f}")
print(f"  OLS R²={summary['ols_r2']:.3f}")
print(f"  RF AUC={summary['rf_auc']:.3f}")
print(f"  Mediation indirect={summary['mediation_indirect']:.4f} sig={summary['mediation_sig']}")
