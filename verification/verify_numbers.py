"""Recomputes the numbers quoted in the manuscript (v18) from the stored results (results_final, results_all_experiments,
results_leave_house_out_all, results_lho_factorial, results_depth_control). Needs pandas, scipy, statsmodels.
Run from anywhere: python3 verify_numbers.py"""
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.formula.api as smf, statsmodels.api as sm
import os
U=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))+os.sep  # nilm project root
K=['appliance','arm','model','imbalance']
ON={"fridge":339857,"washing_machine":35826,"dishwasher":42545,"kettle":5979,"microwave":3632}
wavg=lambda d: sum(d[a]*ON[a] for a in ON)/sum(ON.values())
P0=pd.read_csv(U+'results_final/pooled_final_280.csv'); print('status',P0.status.value_counts().to_dict()); P=P0[P0.status=='ok'].copy()
F1=pd.read_csv(U+'results_final/pooled_first_pass_280.csv'); F1=F1[F1.status=='ok'].copy()
print('n',len(P),len(F1))
print('\n[4.1] acc>f1 all:',(P.accuracy_mean>P.f1_mean).all(),'mean acc %.3f range %.3f-%.3f; mean f1 %.3f range %.3f-%.3f'%(P.accuracy_mean.mean(),P.accuracy_mean.min(),P.accuracy_mean.max(),P.f1_mean.mean(),P.f1_mean.min(),P.f1_mean.max()))
print(P[(P.appliance=='microwave')&(P.arm=='A')&(P.imbalance=='none')][['model','f1_mean','accuracy_mean']].round(4).to_string())
print('n f1==0:',(P.f1_mean==0).sum(), P[P.f1_mean==0][K].values.tolist())
B=P.groupby(['model','appliance']).f1_mean.max().unstack()
print('\n[4.2] best per model mean/sd(ddof1)/sd(ddof0)\n',pd.DataFrame({'mean':B.mean(1),'sd1':B.std(1),'sd0':B.std(1,ddof=0)}).round(3).sort_values('mean',ascending=False))
print('ranks',B.rank(ascending=False).mean(1).round(2).to_dict()); print(B.rank(ascending=False).round(1))
fr=stats.friedmanchisquare(*[B.loc[m] for m in B.index]); print('friedman',fr)
print(B.round(3))
print('\n[4.3] best per model by arm (max over app, imb)'); t=P.groupby(['model','arm']).f1_mean.max().unstack(); t['d']=t.B-t.A; print(t.round(3))
print('arm means',P.groupby('arm').f1_mean.mean().round(4).to_dict(), 'gain %.1f'%(100*(P[P.arm=="B"].f1_mean.mean()/P[P.arm=="A"].f1_mean.mean()-1)))
t=P.groupby(['model','arm','appliance']).f1_mean.max().unstack('appliance').mean(1).unstack(); t['d']=t.B-t.A; print('avg over appliances of best\n',t.round(3))
print('best cfg arm per appliance', P.loc[P.groupby('appliance').f1_mean.idxmax()][K+['f1_mean','f1_std','precision_mean','recall_mean','accuracy_mean','n_rows']].round(4).to_string())
Q=P[P.model!='kNN']
print('\n[4.4] bestcase',Q.groupby(['model','imbalance']).f1_mean.max().unstack().mean().round(3).to_dict()); print('means',Q.groupby('imbalance').f1_mean.mean().round(4).to_dict())
print('LR best by imb',P[P.model=='LogisticRegression'].groupby('imbalance').f1_mean.max().round(3).to_dict())
tb=P[(P.arm=='B')&P.model.isin(['DecisionTree','RandomForest','XGBoost','LightGBM'])].groupby(['appliance','imbalance']).f1_mean.mean().unstack(); print('Table6\n',tb.round(3))
print('\n[4.5] second best diffs'); 
for a,d in P.groupby('appliance'):
    s=d.sort_values('f1_mean',ascending=False); print(a, s.iloc[0][K[1:]].tolist(), round(s.iloc[0].f1_mean,4), s.iloc[1][K[1:]].tolist(), round(s.iloc[1].f1_mean,4), 'diff %.4f'%(s.iloc[0].f1_mean-s.iloc[1].f1_mean), 'range %.3f min %.3f'%(s.f1_mean.max()-s.f1_mean.min(), s.f1_mean.min()), len(d))
fb=P[(P.appliance=='fridge')].groupby('model').f1_mean.max(); print('fridge best per model',fb.round(4).to_dict())
r=P[(P.appliance=='fridge')&(P.model=='RandomForest')&(P.arm=='B')&(P.imbalance=='class_weight')].iloc[0]
n=1080000; on=339857; tp=r.recall_mean*on; fp=tp*(1-r.precision_mean)/r.precision_mean; print('fridge FPR approx %.4f'%(fp/(n-on)))
print('\n[4.6]'); M=P.merge(F1,on=K,suffixes=('','_fp')); rr=M[M.model.isin(['SVM_RBF','RandomForest','XGBoost','LightGBM'])]
print('n rerun',len(rr),'mean dF1 %.4f mean dAcc %.4f'%((rr.f1_mean-rr.f1_mean_fp).mean(),(rr.accuracy_mean-rr.accuracy_mean_fp).mean()))
print(rr.assign(d=rr.f1_mean-rr.f1_mean_fp).groupby('model').d.agg(['mean',lambda x:(x>0).sum(),'size']).round(4))
oth=M[~M.model.isin(['SVM_RBF','RandomForest','XGBoost','LightGBM'])]; print('others unchanged:',np.allclose(oth.f1_mean,oth.f1_mean_fp))
for a in ON:
    d=F1[F1.appliance==a]; b=d.loc[d.f1_mean.idxmax()]; d2=P[P.appliance==a]; b2=d2.loc[d2.f1_mean.idxmax()]; print(a,b[K[1:]].tolist(),round(b.f1_mean,4),'->',b2[K[1:]].tolist(),round(b2.f1_mean,4),'%.4f'%(b2.f1_mean-b.f1_mean))
B1=F1.groupby(['model','appliance']).f1_mean.max().unstack(); print('first-pass ranks',B1.rank(ascending=False).mean(1).round(2).to_dict())
four=['SVM_RBF','RandomForest','XGBoost','LightGBM']; print('friedman 4 rerun final',stats.friedmanchisquare(*[B.loc[m] for m in four]),'first',stats.friedmanchisquare(*[B1.loc[m] for m in four]),'first all7',stats.friedmanchisquare(*[B1.loc[m] for m in B1.index]))
print('\n[4.7] ANOVA main 270')
m=smf.ols('f1_mean ~ C(appliance)+C(model)+C(arm)+C(imbalance)',data=P).fit(); a=sm.stats.anova_lm(m,typ=2); print((100*a.sum_sq/a.sum_sq.sum()).round(2).to_dict()); print(a['PR(>F)'].to_dict()); 
tot=((P.f1_mean-P.f1_mean.mean())**2).sum(); print('share of TOTAL SS',(100*a.sum_sq/tot).round(2).to_dict(),'r2',round(m.rsquared,4))
# cluster-robust? skip
print('\n[4.8] 35 configs'); L35=pd.read_csv(U+'results_leave_house_out_all/fold_results_unified.csv'); print(L35.groupby(['appliance','model']).size().unstack().to_string())
g=pd.read_csv(U+'results_final/lho270_summary.csv')
s35=L35.groupby(['appliance','model','imbalance']).agg(lho=('f1','mean'),sd=('f1','std'),mn=('f1','min'),mx=('f1','max'),p=('precision','mean'),r=('recall','mean')).reset_index()
pb=P[P.arm=='B']; pbm=pb.loc[pb.groupby(['appliance','model']).f1_mean.idxmax()][['appliance','model','imbalance','f1_mean']]
s35=s35.merge(pbm,on=['appliance','model'],suffixes=('','_pool')); print('same imbalance as pooled best FS-B:',(s35.imbalance==s35.imbalance_pool).all(), len(s35))
s35=s35.merge(g[g.arm=='B'][['appliance','model','imbalance','lho']].rename(columns={'lho':'lho270'}),on=['appliance','model','imbalance']); s35['drop']=s35.f1_mean-s35.lho
print('max abs diff 35 vs 270:',(s35.lho-s35.lho270).abs().max()); print(s35[(s35.lho-s35.lho270).abs()>1e-6][['appliance','model','imbalance','lho','lho270']].round(4).to_string())
print('per-model mean drop',s35.groupby('model')['drop'].mean().round(3).to_dict())
pv=s35.pivot(index='appliance',columns='model',values='lho'); pp=s35.pivot(index='appliance',columns='model',values='f1_mean')
print('LHO ranks',pv.rank(axis=1,ascending=False).mean().round(2).to_dict(),'friedman',stats.friedmanchisquare(*[pv[c] for c in pv.columns]))
print('pooled ranks (FS-B best)',pp.rank(axis=1,ascending=False).mean().round(2).to_dict())
print('spearman mean ranks',stats.spearmanr(pv.rank(axis=1,ascending=False).mean(),pp.rank(axis=1,ascending=False).mean()), 'vs overall pooled ranks', stats.spearmanr(pv.rank(axis=1,ascending=False).mean().sort_index(),B.rank(ascending=False).mean(1).sort_index()))
print('dishwasher\n',s35[s35.appliance=='dishwasher'][['model','imbalance','f1_mean','lho','drop']].round(3).to_string())
print('rho pooled vs drop',stats.spearmanr(s35.f1_mean,s35['drop']),'rho mean vs drop',stats.spearmanr((s35.f1_mean+s35.lho)/2,s35['drop']))
best=P.loc[P.groupby('appliance').f1_mean.idxmax()]
t9=s35.merge(best[['appliance','model','imbalance']],on=['appliance','model','imbalance']); print('Table9\n',t9[['appliance','model','imbalance','f1_mean','lho','sd','mn','mx','p','r','drop']].round(3).to_string())
print('w pooled %.4f w lho %.4f macro %.4f %.4f'%(wavg(dict(zip(t9.appliance,t9.f1_mean))),wavg(dict(zip(t9.appliance,t9.lho))),t9.f1_mean.mean(),t9.lho.mean()))
# sequential SS decomposition
L35['grp']=L35.appliance+'_'+L35.fold.astype(str)
m=smf.ols('f1 ~ C(appliance)+C(grp)+C(model)',data=L35).fit(); a=sm.stats.anova_lm(m,typ=1); print('seq SS',(100*a.sum_sq/a.sum_sq.sum()).round(1).to_dict())
print('\n[4.9]'); print('rho',stats.spearmanr(g.pooled,g.lho)[0],'n lower',(g['drop']>0).sum(),'rho drop',stats.spearmanr(g.pooled,g['drop']),'rho mean',stats.spearmanr((g.pooled+g.lho)/2,g['drop']))
w=g.pivot_table(index=['appliance','model','imbalance'],columns='arm',values='lho').dropna(); print('B>A',(w.B>w.A).sum(),len(w),'wilcoxon',stats.wilcoxon(w.B,w.A))
print('arm means lho',g.groupby('arm').lho.mean().round(4).to_dict(),'drop',g.groupby('arm')['drop'].mean().round(4).to_dict())
gq=g[g.model!='kNN']; print('imb lho 6 models',gq.groupby('imbalance').lho.mean().round(4).to_dict(),'pooled',gq.groupby('imbalance').pooled.mean().round(4).to_dict(),'drop',gq.groupby('imbalance')['drop'].mean().round(4).to_dict())
m=smf.ols('lho ~ C(appliance)+C(model)+C(arm)+C(imbalance)',data=g).fit(); a=sm.stats.anova_lm(m,typ=2); print('eta lho',(100*a.sum_sq/a.sum_sq.sum()).round(2).to_dict())
for a_,d in g.groupby('appliance'):
    bp=d.loc[d.pooled.idxmax()]; bl=d.loc[d.lho.idxmax()]; print(a_,bp[K[1:]].tolist(),round(bp.lho,4),bl[K[1:]].tolist(),round(bl.lho,4),'lost %.4f'%(bl.lho-bp.lho))
ns=pd.read_csv(U+'results_final/nested_selection_phaseC.csv'); d=ns.grouped_f1-ns.pooled_f1
print('nested: better',(d>0).sum(),'worse',(d<0).sum(),'tie',(d==0).sum(),'wilcoxon',stats.wilcoxon(ns.grouped_f1,ns.pooled_f1),'wilcoxon pratt',stats.wilcoxon(ns.grouped_f1,ns.pooled_f1,zero_method='pratt'),'mean diff %.4f'%d.mean())
s=ns.groupby('appliance')[['grouped_f1','pooled_f1','oracle_f1']].mean(); print(s.round(4)); print('w',wavg(s.grouped_f1),wavg(s.pooled_f1),wavg(s.oracle_f1), 'wilcoxon on 5 appliances',stats.wilcoxon(s.grouped_f1,s.pooled_f1))
print('corr chosen: grouped',(~ns.grouped_sel.str.endswith('none')).sum(),'pooled',(~ns.pooled_sel.str.endswith('none')).sum())
print('\n[5.3] depth'); D=pd.read_csv(U+'results_depth_control/summary_results.csv'); print(D.status.value_counts().to_dict(), D.model.unique())
x=D.merge(F1[F1.model=='RandomForest'],on=K,suffixes=('_dc','_fp')); x['d']=x.f1_mean_dc-x.f1_mean_fp; print('n',len(x),'unchanged',(x.d.abs()<1e-9).sum(),'changed',(x.d.abs()>=1e-9).sum(),'mean all %.4f'%x.d.mean(),'mean changed %.4f median %.4f'%(x[x.d.abs()>=1e-9].d.mean(),x[x.d.abs()>=1e-9].d.median()))
rf=M[M.model=='RandomForest']; print('RF total %.4f'%((rf.f1_mean-rf.f1_mean_fp).mean()))
print('\n[CNN D]'); Dq=pd.read_csv(U+'results_all_experiments/phase_D/fold_results.csv'); dd=Dq.groupby('appliance').agg(f1=('f1','mean'),p=('precision','mean'),r=('recall','mean')); print(dd.round(3)); print('w',wavg(dd.f1),'macro',dd.f1.mean())
C=pd.read_csv(U+'results_all_experiments/phase_R_CNN/fold_results.csv'); print(C.groupby(['variant','appliance']).agg(f1=('f1','mean'),ep=('best_epoch','mean'),epmax=('best_epoch','max'),thr=('threshold','mean'),ntr=('n_train','mean'),n=('f1','size')).round(3))

print('\n[4.8 new sentence] micro-aggregated leave-house-out F1 (confusion counts pooled over the five folds)')
F=pd.read_csv(U+'results_lho_factorial/fold_results.csv'); A=pd.read_csv(U+'results_all_experiments/phase_A/fold_results.csv'); E=pd.read_csv(U+'results_all_experiments/phase_E/fold_results.csv')
L=pd.concat([F,A,E],ignore_index=True); gg=L.groupby(K).agg(mean_f1=('f1','mean'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum')).reset_index(); gg['micro']=2*gg.tp/(2*gg.tp+gg.fp+gg.fn)
gg=gg.merge(P[K+['f1_mean']],on=K); bb=gg.loc[gg.groupby('appliance').f1_mean.idxmax()].set_index('appliance')
print(bb[['model','arm','imbalance','f1_mean','mean_f1','micro']].round(3)); print('weighted: pooled %.3f, unseen mean of folds %.3f, unseen pooled counts %.3f'%(wavg(bb.f1_mean),wavg(bb.mean_f1),wavg(bb.micro)))
del C
print('\n[S8 fix] main-effects p-values, 270 configurations'); m=smf.ols('f1_mean ~ C(appliance)+C(model)+C(arm)+C(imbalance)',data=P).fit(); print(sm.stats.anova_lm(m,typ=2)['PR(>F)'].to_dict())
