"""Independent S02b calibration replay and zero-denominator report builder."""
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stages/S02b_recovery'
sys.path.insert(0,str(ROOT/'stages/S01_protocol/attempt_02'))
import protocol_core as core
sys.path.insert(0,str(OUT/'scripts'))
import calibrate

def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    if p.exists():
        if p.read_text()!=raw:raise FileExistsError(f'existing output differs: {p}')
        return
    p.write_text(raw)

def independent_detect(dates,values):
    days=np.asarray(dates,dtype='datetime64[D]'); v=np.asarray(values)
    out=[];last=None
    for i in range(len(v)):
        hit=(i>=1 and v[i]-v[i-1]>=7) or (i>=5 and v[i]-v[i-5]>=10)
        if hit and (last is None or int((days[i]-days[last])/np.timedelta64(1,'D'))>30):
            end=min(i+11,len(v));p=i+int(np.argmax(v[i:end]));out.append((i,p));last=i
    return out

def independent_common(v,ev):
    n=0; exclusions={'argmax':{},'onset5':{}}
    for j,(on,peak) in enumerate(ev):
        valid=[];next_on=ev[j+1][0] if j+1<len(ev) else len(v)
        for name,a in (('argmax',peak),('onset5',on+5)):
            reason=None
            if on<250:reason='prehistory'
            elif a+120>=len(v):reason='end_of_data'
            elif a+120>=next_on:reason='next_onset'
            else:
                pre=np.log(v[on-250:on]); den=np.log(v[a])-np.mean(pre)
                if not np.isfinite(den) or den<=1e-8:reason='nonpositive normalization denominator'
            if reason:exclusions[name][reason]=exclusions[name].get(reason,0)+1
            valid.append(reason is None)
        n+=int(all(valid))
    return n,exclusions

def main():
    freeze=json.loads((OUT/'calibration_freeze.json').read_text())
    for path,sha in freeze['files'].items():assert h(ROOT/path)==sha,(path,'freeze hash mismatch')
    p=json.loads((OUT/'calibration_parameters.json').read_text())
    rows=list(csv.DictReader((OUT/'calibration_diagnostics.csv').open()))
    assert len(rows)==33
    s=pd.read_csv(ROOT/'omori_pipeline/data/vix_term_structure.csv',parse_dates=['Date']).dropna(subset=['VIX30D'])
    dates=s.Date.to_numpy(); v=s.VIX30D.to_numpy();x=np.log(v)
    obs=np.array([x.mean(),np.sqrt(np.mean((x-x.mean())**2))]+[np.corrcoef(x[:-lag],x[lag:])[0,1] for lag in (1,22,63)])
    assert np.max(np.abs(obs-p['observed_moments']))<1e-12
    empirical=independent_detect(dates,v);emp_common,emp_exclusions=independent_common(v,empirical)
    assert len(empirical)==p['empirical_raw_detected']==34
    assert emp_common==p['empirical_common_eligible']==17
    alignment={'empirical':{'raw_detected':len(empirical),'common_eligible':emp_common,'named_merged_catalog':33,'anchor_exclusions':emp_exclusions},'processes':{},'alignment_scope':'raw detector, 120 observed sessions, paired argmax/onset5 anchors; no confirmatory recovery'}
    checks=[]
    for proc in calibrate.PROCESSES:
        info=p['processes'][proc];q=info['parameters'];rproc=[r for r in rows if r['process']==proc]
        assert len(rproc)==11
        vals=[];counts=[];elig=[]
        if proc=='ARFIMA':
            cov=core.arfima_cov(len(v),q[0],obs[1]);circ=np.r_[cov,cov[-2:0:-1]]
            eig=np.fft.fft(circ).real
            assert eig.min()>=-1e-10
            assert np.max(np.abs(np.fft.ifft(eig).real[:64]-cov[:64]))<1e-10
        for i,row in enumerate(rproc):
            assert int(row['path_id'])==i
            seed=core.seed('S02b','calibration',proc,i)
            assert int(row['seed'])==seed
            path=calibrate.generate(proc,q,obs[0],obs[1],len(v),seed)
            measures=np.array([path.mean(),np.sqrt(np.mean((path-path.mean())**2))]+[np.corrcoef(path[:-lag],path[lag:])[0,1] for lag in (1,22,63)])
            assert np.max(np.abs(measures-[float(row[k]) for k in ('mean','sd','acf1','acf22','acf63')]))<1e-10
            ev=independent_detect(dates,np.exp(path));common,exclusions=independent_common(np.exp(path),ev)
            assert len(ev)==int(row['detected']) and common==int(row['eligible_common'])
            vals.append(measures);counts.append(len(ev));elig.append(common)
        med=np.median(vals,axis=0)
        assert np.max(np.abs(med-info['median_generated_moments']))<1e-10
        assert int(np.median(counts))==info['median_detected']
        assert int(np.median(elig))==info['median_eligible_common']
        acf=np.array([np.prod([(j-1+q[0])/(j-q[0]) for j in range(1,k+1)]) for k in (1,22,63)]) if proc=='ARFIMA' else calibrate.theory(proc,q)
        assert np.max(np.abs(acf-info['theory_acf']))<1e-10
        theory_pass=bool(np.all(np.abs(acf-obs[2:])<=calibrate.ACF_TOL))
        assert theory_pass==info['gates']['theory_acf']
        assert info['status']=='NOT CALIBRATABLE'
        alignment['processes'][proc]={'status':info['status'],'median_raw_detected':int(np.median(counts)),'median_common_eligible':int(np.median(elig)),
            'path_raw_detected':counts,'path_common_eligible':elig,'gates':info['gates'],'parameters':q,
            'theory_acf_residual':info['theory_acf_residual'],'median_generated_moment_residual':(med-obs).tolist()}
        checks.append(dict(process=proc,paths_replayed=11,detector_parity=True,eligibility_parity=True,moment_parity=True,theoretical_acf_parity=True))
    save('alignment_results.json',alignment)
    header=['true_process','anchor','start','status','planned_n','completed_n','failed_n','eligible_n','denominator_n','winner','count','rate','wilson_low','wilson_high','undetermined_share','reason']
    with (OUT/'recovery_rates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=header);w.writeheader()
        for proc in calibrate.PROCESSES:
            for anchor in ('argmax','onset5'):
                for start in (1,5):
                    for winner in ('E1','E2','PL','SE','undetermined'):
                        w.writerow(dict(true_process=proc,anchor=anchor,start=start,status='NOT CALIBRATABLE',planned_n=0,completed_n=0,failed_n=0,eligible_n=0,denominator_n=0,winner=winner,count='',rate='',wilson_low='',wilson_high='',undetermined_share='',reason='calibration_gates_failed'))
    with (OUT/'confusion_matrix.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['true_process','anchor','start','winner','count','denominator_n','proportion'])
    with (OUT/'simulation_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['process','cell','replication_id','seed','status'])
    with (OUT/'failures.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['process','phase','status','failed_gates'])
        for proc in calibrate.PROCESSES:
            gates=p['processes'][proc]['gates'];w.writerow([proc,'calibration','NOT CALIBRATABLE',';'.join(k for k,v in gates.items() if not v)])
    save('allocation_spec.json',{'status':'NO_ELIGIBLE_PROCESSES','rule':'only processes passing all prespecified calibration gates enter recovery','calibration_path_count_per_process':11,'retained_processes':[],'confirmatory_cells':[],'planned_n_per_cell':0,'completed_n_per_cell':0,'planned_confirmatory_path_ids':[],'reason':'all three processes NOT CALIBRATABLE; no point-score outcomes opened','historical_s02a_n':0,'historical_pilots_excluded':True})
    save('independent_verification.json',{'status':'PASS_FOR_CALIBRATION_AND_EMPTY_RECOVERY_SCOPE','checks':checks,'empirical_raw_detected':34,'empirical_common_eligible':17,'frozen_file_hashes_pass':len(freeze['files']),'calibration_rows':len(rows),'confirmatory_replications':0,'zero_denominator_rates_blank':True})
    snap={name:h(OUT/name) for name in ('calibration_parameters.json','calibration_diagnostics.csv','detector_parity.csv','alignment_results.json','recovery_rates.csv','confusion_matrix.csv','simulation_manifest.csv','failures.csv','allocation_spec.json','independent_verification.json')}
    save('checkpoints/checkpoint_000_calibration_complete.json',{'checkpoint_id':'000','immutable':True,'phase':'calibration_complete','calibration_paths_completed':33,'calibration_paths_planned':33,'confirmatory_planned':0,'confirmatory_completed':0,'confirmatory_failed':0,'confirmatory_pending':0,'status':'PARTIAL_RECOVERY_NOT_ESTIMABLE','outputs_sha256':snap,'original_s02b_units_reused':0,'historical_s02a_pilots_reused_as_outcomes':0})
    print(json.dumps({'independent_checks':'PASS','calibration_paths':33,'eligible_processes':0,'confirmatory_n':0}))

if __name__=='__main__':main()
