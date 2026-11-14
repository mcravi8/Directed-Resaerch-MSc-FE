"""Read-only pre-staging inventory, secret-pattern and source-copy checks."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
patterns = {
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'GitHub token': r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}',
    'AWS access key': r'AKIA[0-9A-Z]{16}',
    'credential URL': r'https?://[^\s/:]+:[^\s/@]+@',
    'assigned secret': r'''(?i)(?:api_key|api_secret|password|access_token)\s*=\s*['"][^'"\s]{12,}['"]''',
    'personal identifier': r'(?i)(?:student\s*id|usc\s*id)\D{0,12}\d{8,}',
}
issues=[]; count=0; python_count=0
for p in ROOT.rglob('*'):
    if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in ('.git','.venv','__pycache__','.cache','.tmp','.pytest_cache')): continue
    rel=str(p.relative_to(ROOT));count+=1
    if p.suffix in {'.pdf','.doc','.docx','.zip','.pem','.key','.p12','.pfx','.log'} or any(x in p.parts for x in ('raw','private_sources','user_extract','S05_release')):
        issues.append((rel,'excluded file type or directory'))
    if p.is_symlink(): issues.append((rel,'symbolic link'))
    try: text=p.read_text()
    except UnicodeDecodeError:
        issues.append((rel,'unreviewed binary'));continue
    for label,pat in patterns.items():
        if re.search(pat,text):issues.append((rel,label))
    if p.suffix=='.py':
        ast.parse(text,filename=rel);python_count+=1
for row in json.loads((ROOT/'SOURCE_MAP.json').read_text()):
    p=ROOT/row['publication']
    if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:issues.append((row['publication'],'source-copy mismatch'))
print(json.dumps({'files':count,'python_files_parsed':python_count,'issues':issues},indent=2))
raise SystemExit(bool(issues))
