"""Read-only source checks; never imports project code or starts ROS/training."""
import ast
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
errors = []
counts = dict(python=0, shell=0, xml=0)
for path in ROOT.rglob('*'):
    if not path.is_file() or '.git' in path.relative_to(ROOT).parts:
        continue
    relative = str(path.relative_to(ROOT))
    try:
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=relative)
            counts['python'] += 1
        elif path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if result.returncode:
                raise ValueError(result.stderr.strip())
            counts['shell'] += 1
        elif path.suffix in {'.xml', '.xacro', '.sdf', '.dae'}:
            ET.parse(path)
            counts['xml'] += 1
        if path.name == '.env' or path.suffix in {'.pt','.pth','.safetensors','.npz','.npy','.mp4','.mov','.key','.pem'}:
            raise ValueError('Unexpected local/generated artifact')
    except Exception as exc:
        errors.append(f'{relative}: {exc}')
print(counts)
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print('PASS: source checks only; no environment or runtime validation implied.')
