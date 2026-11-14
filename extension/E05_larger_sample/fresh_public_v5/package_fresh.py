"""Archive complete checkpoint units and hash the reproducible public handoff."""
import hashlib,json,tarfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def info(path):return dict(bytes=path.stat().st_size,sha256=sha(path))
if __name__=='__main__':
    assert (HERE/'public_run_end.json').exists() and (HERE/'pooled_summary.json').exists()
    public=json.loads((HERE/'summary_status.json').read_text())
    pooled=json.loads((HERE/'pooled_summary.json').read_text())
    assert public['point_panels']==12 and public['saved_draw_units']==11988
    assert pooled['saved_draws']==999
    archives=HERE/'archives';archives.mkdir(exist_ok=True)
    manifest={}
    for panel in sorted(p for p in (HERE/'units').iterdir() if p.is_dir()):
        units=sorted(panel.glob('*.json'))
        assert len(units)==1000 and (panel/'point.json').exists()
        assert {f'{i:04d}.json' for i in range(999)} <= {p.name for p in units}
        target=archives/(panel.name+'.tar.gz')
        with tarfile.open(target,'w:gz') as tar:
            for path in units:tar.add(path,arcname=panel.name+'/'+path.name)
        manifest[str(target.relative_to(HERE))]=dict(**info(target),units=len(units))
    units=[HERE/'units/point.json',*[HERE/'units'/f'{i:04d}.json' for i in range(999)]]
    assert all(p.exists() for p in units)
    target=archives/'equal_index_pool.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for path in units:tar.add(path,arcname='equal_index_pool/'+path.name)
    manifest[str(target.relative_to(HERE))]=dict(**info(target),units=len(units))
    (HERE/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name!='output_manifest.json' and path.suffix!='.tmp':
            files[path.name]=info(path)
    out=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),archives=manifest,files=files,
        raw_scalar_sources_distributed=False,public_draw_units=public['saved_draw_units'],pooled_draws=pooled['saved_draws'])
    (HERE/'output_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(archives=len(manifest),files=len(files),public_draw_units=public['saved_draw_units'],pooled_draws=pooled['saved_draws'])))
