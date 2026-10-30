"""Leave-house-out analysis behind Sections 4.7 to 4.10: effects, variance decomposition, selection bias,
repeated partitions, house-grouped tuning and the CNN baseline. Run after consolidate_results.py.
Outputs in results_final/: analysis_unseen_houses.json, lho270_summary.csv, anova_pooled_270.json"""
import os, pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.formula.api as smf, statsmodels.api as sm
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); U=os.path.join(ROOT,'results_all_experiments')+'/'; OUT=os.path.join(ROOT,'results_final')
P=pd.read_csv(os.path.join(OUT,'pooled_final_280.csv')); P=P[P.status=='ok'][['appliance','arm','model','imbalance','f1_mean']].rename(columns={'f1_mean':'pooled'})
F=pd.read_csv(os.path.join(ROOT,'results_lho_factorial','fold_results.csv')); F['seed']=42
A=pd.read_csv(U+'phase_A/fold_results.csv'); E=pd.read_csv(U+'phase_E/fold_results.csv'); Bq=pd.read_csv(U+'phase_B/fold_results.csv')
Cq=pd.read_csv(U+'phase_C/fold_results.csv'); Dq=pd.read_csv(U+'phase_D/fold_results.csv')
K=['appliance','arm','model','imbalance']
L=pd.concat([F,A,E],ignore_index=True)
g=L.groupby(K).agg(lho=('f1','mean'),n=('f1','size'),on=('n_test_on','sum')).reset_index().merge(P,on=K); g['drop']=g.pooled-g.lho
R={'n_cfg':len(g),'n5':int((g.n==5).sum())}
def effects(d,col):
    a=d.groupby('arm')[col].mean(); i=d.groupby('imbalance')[col].mean()
    return dict(A=a.A,B=a.B,gain=100*(a.B/a.A-1),none=i['none'],cw=i['class_weight'],smote=i['SMOTE'],adasyn=i['ADASYN'],cw_gain=100*(i['class_weight']/i['none']-1),corr_gain=100*(i[['class_weight','SMOTE','ADASYN']].mean()/i['none']-1))
R['eff_pooled']=effects(g,'pooled'); R['eff_lho']=effects(g,'lho')
w=g.pivot_table(index=['appliance','model','imbalance'],columns='arm',values='lho').dropna(); R['BgtA']=[int((w.B>w.A).sum()),len(w)]
wp=g.pivot_table(index=['appliance','model','imbalance'],columns='arm',values='pooled').dropna(); R['BgtA_pooled']=[int((wp.B>wp.A).sum()),len(wp)]
R['drop_arm']=g.groupby('arm')['drop'].mean().to_dict(); R['drop_imb']=g.groupby('imbalance')['drop'].mean().to_dict(); R['drop_model']=g.groupby('model')['drop'].mean().to_dict()
R['app_gain']={a:[d[d.arm=='B'].pooled.mean()-d[d.arm=='A'].pooled.mean(), d[d.arm=='B'].lho.mean()-d[d.arm=='A'].lho.mean()] for a,d in g.groupby('appliance')}
eta={}
for col in ['pooled','lho']:
    m=smf.ols(f'{col} ~ C(appliance)+C(model)+C(arm)+C(imbalance)',data=g).fit(); a=sm.stats.anova_lm(m,typ=2); eta[col]=(100*a.sum_sq/a.sum_sq.sum()).to_dict()
R['eta']=eta
R['rho_all']=stats.spearmanr(g.pooled,g.lho)[0]; R['rho_opt']=list(stats.spearmanr(g.pooled,g['drop'])); R['n_lower']=int((g['drop']>0).sum())
sel={}
for a,d in g.groupby('appliance'):
    bp=d.loc[d.pooled.idxmax()]; bl=d.loc[d.lho.idxmax()]
    sel[a]=dict(p=[bp.model,bp.arm,bp.imbalance],p_pooled=bp.pooled,p_lho=bp.lho,l=[bl.model,bl.arm,bl.imbalance],l_pooled=bl.pooled,l_lho=bl.lho)
R['sel']=sel; R['n_changed']=sum(v['p']!=v['l'] for v in sel.values())
# weighted F1 of selections
onA=g.groupby('appliance').on.first()
def wsum(vals): return sum(vals[a]*onA[a] for a in vals)/sum(onA[a] for a in vals)
R['w_pooledsel']=wsum({a:v['p_lho'] for a,v in sel.items()}); R['w_lhobest']=wsum({a:v['l_lho'] for a,v in sel.items()})
# models: pooled-selected per model per appliance, LHO score; friedman
pm=g.loc[g.groupby(['appliance','model']).pooled.idxmax()].pivot(index='appliance',columns='model',values='lho')
lm=g.groupby(['appliance','model']).lho.max().unstack()
for nm,t in [('fried_pooledsel',pm),('fried_lhobest',lm)]:
    r=stats.friedmanchisquare(*[t[c] for c in t.columns]); R[nm]=[r.statistic,r.pvalue]; R[nm+'_ranks']=t.rank(axis=1,ascending=False).mean().round(2).to_dict()
pmP=g.loc[g.groupby(['appliance','model']).pooled.idxmax()].pivot(index='appliance',columns='model',values='pooled'); R['fried_pooled']=list(stats.friedmanchisquare(*[pmP[c] for c in pmP.columns]))
R['lm']=lm.round(3).to_dict()
g.to_csv(os.path.join(OUT,'lho270_summary.csv'),index=False)
# ---- Phase B: repeated partitions (4 fast models)
fast=['LogisticRegression','DecisionTree','XGBoost','LightGBM']
B=pd.concat([Bq,L[L.model.isin(fast)]],ignore_index=True)
rows=[]
for s,d in B.groupby('seed'):
    h=d.groupby(K).agg(lho=('f1','mean'),on=('n_test_on','sum')).reset_index().merge(P,on=K)
    e=effects(h,'lho'); ch=0; loss=[]; wv={}
    for a,x in h.groupby('appliance'):
        bp=x.loc[x.pooled.idxmax()]; bl=x.loc[x.lho.idxmax()]; ch+= int((bp.model,bp.arm,bp.imbalance)!=(bl.model,bl.arm,bl.imbalance)); loss.append(bl.lho-bp.lho); wv[a]=bp.lho
    m=smf.ols('lho ~ C(appliance)+C(model)+C(arm)+C(imbalance)',data=h).fit(); a_=sm.stats.anova_lm(m,typ=2); et=(100*a_.sum_sq/a_.sum_sq.sum())
    rows.append(dict(seed=s,n=len(h),gain=e['gain'],cw_gain=e['cw_gain'],corr_gain=e['corr_gain'],changed=ch,max_loss=max(loss),mean_lho=h.lho.mean(),w_sel=wsum(wv),eta_model=et['C(model)'],eta_imb=et['C(imbalance)'],eta_arm=et['C(arm)'],eta_app=et['C(appliance)'],rho=stats.spearmanr(h.pooled,h.lho)[0]))
Rb=pd.DataFrame(rows); R['B']=Rb.round(4).to_dict('records'); R['B_mean']=Rb.mean().round(4).to_dict(); R['B_sd']=Rb.std().round(4).to_dict()
print(Rb.round(3).to_string())
# ---- Phase C: house-grouped tuning vs fixed
c=Cq.groupby(['appliance','model','imbalance']).f1.mean().rename('tuned').reset_index()
fx=g[g.arm=='B'][['appliance','model','imbalance','lho','pooled']]; c=c.merge(fx,on=['appliance','model','imbalance'])
R['C']=dict(n=len(c),mean_tuned=c.tuned.mean(),mean_fixed=c.lho.mean(),diff=(c.tuned-c.lho).mean(),absdiff=(c.tuned-c.lho).abs().mean(),rho=stats.spearmanr(c.tuned,c.lho)[0],
  cw_gain_tuned=100*(c[c.imbalance=='class_weight'].tuned.mean()/c[c.imbalance=='none'].tuned.mean()-1),cw_gain_fixed=100*(c[c.imbalance=='class_weight'].lho.mean()/c[c.imbalance=='none'].lho.mean()-1),
  corr_gain_tuned=100*(c[c.imbalance!='none'].tuned.mean()/c[c.imbalance=='none'].tuned.mean()-1))
bt={a:d.loc[d.tuned.idxmax(),['model','imbalance','tuned']].tolist() for a,d in c.groupby('appliance')}; R['C_best']=bt
R['C_none_best_cnt']=sum(1 for v in bt.values() if v[1]=='none')
# ---- Phase D: CNN
dd=Dq.groupby('appliance').agg(f1=('f1','mean'),sd=('f1','std'),on=('n_test_on','sum'),p=('precision','mean'),r=('recall','mean'))
R['D']=dd.round(3).to_dict('index'); R['D_w']=float((dd.f1*dd.on).sum()/dd.on.sum()); R['D_macro']=float(dd.f1.mean())
R['D_vs']={a:dict(cnn=dd.f1[a],classical_pooledsel=sel[a]['p_lho'],classical_lhobest=sel[a]['l_lho']) for a in dd.index}
m2=smf.ols('pooled ~ (C(appliance)+C(model)+C(arm)+C(imbalance))**2',data=P).fit(); a2=sm.stats.anova_lm(m2,typ=2)
m1=smf.ols('pooled ~ C(appliance)+C(model)+C(arm)+C(imbalance)',data=P).fit(); a1=sm.stats.anova_lm(m1,typ=2)
json.dump({'main':(100*a1.sum_sq/a1.sum_sq.sum()).to_dict(),'twoway':(100*a2.sum_sq/a2.sum_sq.sum()).to_dict(),'p':a2['PR(>F)'].dropna().to_dict(),'r2':m2.rsquared},open(os.path.join(OUT,'anova_pooled_270.json'),'w'),indent=1)
json.dump(R,open(os.path.join(OUT,'analysis_unseen_houses.json'),'w'),indent=1,default=float)
print(json.dumps({k:R[k] for k in ['n_cfg','n5','eff_pooled','eff_lho','BgtA','BgtA_pooled','drop_imb','eta','rho_all','rho_opt','n_lower','n_changed','w_pooledsel','w_lhobest','fried_pooledsel','fried_lhobest','fried_pooled','fried_pooledsel_ranks','C','C_best','D_w','D_macro','D_vs']},indent=0,default=lambda x: round(float(x),3)))
for a,v in sel.items(): print(a,v['p'],round(v['p_pooled'],3),round(v['p_lho'],3),'|',v['l'],round(v['l_lho'],3))
print(R['app_gain'])
