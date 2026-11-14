"""Independent audit of archived E2 units and published CSVs; no E2 imports."""
import collections
import csv
import hashlib
import io
import json
import math
import pathlib
import sys
import zipfile

P = pathlib.Path(__file__).resolve().parent
TRUTHS = ('E1', 'E2', 'PL', 'SE')
NS = (19, 40, 80, 160, 320)
DESIGNS = ('independent', 'calendar_joint')
Z = 1.959963984540054
issues = []
counts = collections.Counter()
units = collections.Counter()
bound = collections.Counter()
ties = collections.Counter()
fails = collections.Counter()
draw_counts = collections.Counter()
distinct_counts = collections.Counter()

def check(ok, label):
    if not ok:
        issues.append(label)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def rows(name):
    with (P / name).open(newline='') as f:
        return list(csv.DictReader(f))

def wilson(k, n=200):
    p = k/n
    den = 1+Z*Z/n
    mid = (p+Z*Z/(2*n))/den
    half = Z*math.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/den
    return mid-half, mid+half

spec = (P / 'frozen_spec.md').read_bytes()
check(sha(spec) == sha((P.parent / 'E01_protocol/E2_spec_v2.md').read_bytes()), 'E2 spec differs from E1 freeze')
check((P/'spec.sha256').read_text().split()[0] == sha(spec), 'spec.sha256 mismatch')
manifest = json.loads((P/'input_manifest.json').read_text())
for path, meta in manifest['files'].items():
    data=(P.parent.parent/path).read_bytes()
    check(len(data)==meta['bytes'] and sha(data)==meta['sha256'],f'input manifest {path}')
root=P.parent.parent
for line in (root/'stages/S05_release/manifest.sha256').read_text().splitlines():
    digest,filename=line.split(maxsplit=1)
    check(sha((root/'stages/S05_release'/filename).read_bytes())==digest,f'protected release {filename}')
def seed(*items):
    raw=json.dumps([5902026,*items],separators=(',',':'),ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')
parts = collections.defaultdict(list)
for r in rows('archive_parts_manifest.csv'):
    parts[r['archive']].append(r)
archives = rows('archive_manifest.csv')
check(len(archives) == 40 and len(parts) == 40 and sum(map(len,parts.values())) == 68, 'archive coverage mismatch')
for a in archives:
    key = (a['truth'],int(a['N']),a['design'])
    name = pathlib.Path(a['archive']).name
    ps = sorted(parts[name], key=lambda x:int(x['part']))
    check([int(r['part']) for r in ps] == list(range(int(ps[0]['parts']))), f'part indexes {name}')
    chunks=[]
    for r in ps:
        data=(P/r['path']).read_bytes()
        check(len(data)==int(r['bytes']) and sha(data)==r['sha256'],f'part hash {r["path"]}')
        chunks.append(data)
    data=b''.join(chunks)
    check(len(data)==int(a['bytes']) and sha(data)==a['sha256'],f'archive hash {name}')
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        check(zf.testzip() is None,f'zip CRC {name}')
        names=zf.namelist()
        check(len(names)==200,f'zip unit count {name}')
        for nm in names:
            u=json.loads(zf.read(nm))
            rep=int(u['rep']); truth,N,design=key
            check((u['truth'],u['N'],u['design'])==key and 0<=rep<200 and u['unit_id']==f'{truth}/{N}/{design}/{rep}' and nm.endswith(f'/{rep:03d}.json'),f'unit identity {name}:{rep}')
            check(u['status']=='complete' and u['spec_hash']==sha(spec) and u['code_hash']==manifest['files']['extension/E02_sample_size/code/e2.py']['sha256'] and u['input_manifest_hash']==sha((P/'input_manifest.json').read_bytes()),f'unit hashes/status {name}:{rep}')
            check(u['panel_seed']==seed('Extension','E2',truth,N,design,rep,'panel') and u['audit_seed']==seed('Extension','E2',truth,N,design,rep,'boundary_audit'),f'panel/audit seeds {name}:{rep}')
            units[key,rep]+=1
            check(set(u['starts'])=={'1','5'},f'paired starts {name}:{rep}')
            check(len(u['draws']['source_rows'])==N if design=='independent' else len(u['draws'])==u['component_draw_count'],f'draw count {name}:{rep}')
            if design=='calendar_joint':
                selected=[e for d in u['draws'] for e in d['selected_ids']]
                check(len(selected)==N and not u['boundaries'],f'calendar draw/boundary {name}:{rep}')
                distinct=len({d['source_component'] for d in u['draws']})
                check(distinct==u['distinct_source_component_count'],f'distinct components {name}:{rep}')
                draw_counts[key,u['component_draw_count']]+=1
                distinct_counts[key,distinct]+=1
            else:
                check(u['component_draw_count']==N and u['distinct_source_component_count'] is None and not u['boundaries'],f'independent draws {name}:{rep}')
            for start in (1,5):
                case=u['starts'][str(start)]
                check(case['fit_seed']==seed('Extension','E2',truth,N,design,rep,'fit',start),f'fit seed {name}:{rep}:{start}')
                check(len(case['models'])==4,f'families {name}:{rep}:{start}')
                scores={}
                for family in TRUTHS:
                    m=case['models'][family]
                    check(len(m['starts'])==32 and m['n']==121-start and m['k']=={'E1':3,'E2':5,'PL':4,'SE':4}[family],f'fit settings {name}:{rep}:{start}:{family}')
                    best=m['best']
                    if best is None:
                        scores[family]=None; fails[key,start,family]+=1
                    else:
                        sse=best['sse']; score=(121-start)*math.log(sse/(121-start))+2*m['k'] if sse>0 else -math.inf
                        check(math.isclose(score,m['score'],rel_tol=1e-12,abs_tol=1e-9),f'AIC {name}:{rep}:{start}:{family}')
                        scores[family]=m['score']
                        bound[key,start,family]+=int(any(best['active_mask']))
                if any(v is None for v in scores.values()):
                    winner='unclassified_fit_failure'; tied=[]
                else:
                    minimum=min(scores.values());tied=[f for f in TRUTHS if scores[f]==minimum];winner=tied[0]
                check(case['winner']==winner and case['ties']==tied,f'winner/ties {name}:{rep}:{start}')
                ties[key,start]+=len(tied)>1
                counts[key,start,winner]+=1
    check(sorted(rep for k,rep in units if k==key)==list(range(200)),f'replication coverage {name}')

def val(row,col,expected,label):
    v=row[col]
    if isinstance(expected,float):
        check(v!='' and math.isclose(float(v),expected,rel_tol=1e-12,abs_tol=1e-12),f'{label}:{col}: {v} != {expected}')
    else:
        check(v==str(expected),f'{label}:{col}: {v} != {expected}')

checks=collections.Counter()
for r in rows('confusion.csv'):
    key=(r['truth'],int(r['N']),r['design']); s=int(r['start']);w=r['winner'];label=f'confusion {key}/{s}/{w}'
    for c,v in {'count':counts[key,s,w],'completed':200,'planned':200,'pending':0}.items():val(r,c,v,label);checks['confusion']+=1
for r in rows('recognition.csv'):
    key=(r['truth'],int(r['N']),r['design']);s=int(r['start']);k=counts[key,s,r['truth']];lo,hi=wilson(k);label=f'recognition {key}/{s}'
    for c,v in {'correct':k,'rate':k/200,'wilson_lower':lo,'wilson_upper':hi,'completed':200,'planned':200,'unclassified':counts[key,s,'unclassified_fit_failure'],'conditional_denominator':200,'conditional_rate':k/200}.items():val(r,c,v,label);checks['recognition']+=1
for file in ('PL_errors.csv','N80_components.csv'):
    for r in rows(file):
        key=(r['truth'],int(r['N']),r['design']);s=int(r['start']);k=counts[key,s,'PL'];k=200-k if r['estimand']=='PL_miss' else k;lo,hi=wilson(k);label=f'{file} {key}/{s}/{r["estimand"]}'
        for c,v in {'count':k,'rate':k/200,'wilson_lower':lo,'wilson_upper':hi,'completed':200,'planned':200,'unclassified':counts[key,s,'unclassified_fit_failure']}.items():val(r,c,v,label);checks[file]+=1
for r in rows('fit_diagnostics.csv'):
    key=(r['truth'],int(r['N']),r['design']);s=int(r['start']);f=r['family'];label=f'fit diagnostics {key}/{s}/{f}'
    for c,v in {'cases':200,'fit_failures':fails[key,s,f],'bound_contacts':bound[key,s,f],'ties':ties[key,s]}.items():val(r,c,v,label);checks['fit_diagnostics']+=1

qualifying={}
for design in DESIGNS:
    qualifying[design]=[]
    for N in NS:
        passed=all(counts[('PL',N,design),s,'PL']>=160 and all(counts[(t,N,design),s,'PL']<=20 for t in ('E1','E2','SE')) for s in (1,5))
        if passed:qualifying[design].append(N)
for r in rows('N80.csv'):
    d=r['design'];check(r['N80']==(str(min(qualifying[d])) if qualifying[d] else 'not attained on tested grid through 320'),f'N80 {d}')
check(len(rows('confusion.csv'))==400 and len(rows('recognition.csv'))==80 and len(rows('PL_errors.csv'))==80 and len(rows('N80_components.csv'))==80 and len(rows('fit_diagnostics.csv'))==320,'summary row counts')
for r in rows('N80_start_diagnostics.csv'):
    d=r['design'];s=int(r['start'])
    passing=[N for N in NS if counts[('PL',N,d),s,'PL']>=160 and all(counts[(t,N,d),s,'PL']<=20 for t in ('E1','E2','SE'))]
    check(r['secondary_N80']==(str(min(passing)) if passing else 'not attained on tested grid through 320'),f'start threshold {d}/{s}')
for r in rows('component_distribution.csv'):
    key=(r['truth'],int(r['N']),r['design']);n=int(r['N'])
    for col,v in {'replications':200,'component_draw_min':n,'component_draw_mean':float(n),'component_draw_max':n}.items():val(r,col,v,f'components {key}')
    if key[2]=='calendar_joint':
        allx=[x for (cell,x),k in distinct_counts.items() if cell==key for _ in range(k)]
        for col,v in {'distinct_source_min':min(allx),'distinct_source_mean':sum(allx)/200,'distinct_source_max':max(allx)}.items():val(r,col,v,f'components {key}')

draft=(P/'lyra_findings_draft.md').read_text()
import re
table=re.findall(r'^\| (calendar_joint|independent) \| ([15]) \| (19|40|80|160|320) \| (\d+)/200 \| ([\d.]+)% \| ([\d.]+)–([\d.]+)% \|$',draft,re.M)
check(len(table)==20,'Lyra recognition table row count')
for d,s,n,k,rate,lo,hi in table:
    actual=counts[('PL',int(n),d),int(s),'PL']; low,high=wilson(actual)
    check(int(k)==actual and math.isclose(float(rate),round(100*actual/200,1)) and math.isclose(float(lo),round(low*100,1)) and math.isclose(float(hi),round(high*100,1)),f'Lyra recognition {d}/{s}/{n}')
false_table=re.findall(r'^\| (calendar_joint|independent) \| ([15]) \| (\d+)/200 \| (\d+)/200 \| (\d+)/200 \|$',draft,re.M)
check(len(false_table)==4,'Lyra false PL table row count')
for d,s,e1,e2,se in false_table:
    check([int(e1),int(e2),int(se)]==[counts[(t,320,d),int(s),'PL'] for t in ('E1','E2','SE')],f'Lyra false PL {d}/{s}')
inventory=rows('lyra_number_inventory.csv')
check(len(inventory)==201,'Lyra inventory row count')
source_maps={file:rows(file) for file in ('recognition.csv','PL_errors.csv')}
for i,r in enumerate(inventory):
    if r['source_file'] not in source_maps:continue
    criteria=dict(x.split('=',1) for x in r['source_key'].split(';') if '=' in x)
    matches=[x for x in source_maps[r['source_file']] if all(x.get(k)==v for k,v in criteria.items())]
    if len(matches)==1 and r['field'] in matches[0]:
        check(r['value']==matches[0][r['field']],f'Lyra inventory row {i+2}')

out={'issues':issues,'archives':len(archives),'parts':sum(map(len,parts.values())),'units':len(units),'fits':sum(counts.values()),'checks':dict(checks),'qualifying_N':qualifying,'primary_PL_wins':[counts[('PL',N,'calendar_joint'),5,'PL'] for N in NS], 'bound_contacts':sum(bound.values()),'fit_failures':sum(fails.values()),'ties':sum(ties.values()),'confusion':{f'{t},{N},{d},{s},{w}':counts[(t,N,d),s,w] for t in TRUTHS for N in NS for d in DESIGNS for s in (1,5) for w in (*TRUTHS,'unclassified_fit_failure')}}
(P/'vera_verified_inventory.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='confusion'},indent=2))
sys.exit(bool(issues))
