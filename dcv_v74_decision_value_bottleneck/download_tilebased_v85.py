"""Download pinned upstream NPZ splits with SHA256 verification; offline reuse supported."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from tilebased_core_v85 import UPSTREAM_REV

SHA256 = {
 'pkmn/train': '56a8c70a0ecac02c84bff5f9e63190df3a276cc48f4bd2b9b5ca3d9f1fdb1529',
 'pkmn/val': '8903ff2aa1f9afef19cd89211eb7e6404511a1ece00687dcff67e0cae073ed18',
 'pkmn/test': '5ab51e16d9831ebd56fe4f6cec0209fe755a40345e3f6c530b4e554f595d77be',
 'warcraft/train': 'bc1bcb6bbb0be024d1c4f688afa25be7572b9f60f37ed8c45ad56261ed6c87c6',
 'warcraft/val': '13cbedd7cddbaab152be242f4d3a6c96f08cb6f2fdbc8b101e6fe9f7e26d13b0',
 'warcraft/test': 'a0e81856f1a2c749ea6557a407f48eb3b53b7d46d60a7dbbb18cdcf60641b3f6',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(root, datasets, splits, offline=False):
    records = []
    for dataset in datasets:
        for split in splits:
            key = f'{dataset}/{split}'
            path = Path(root) / (key + '.npz')
            expected = SHA256[key]
            url = f'https://raw.githubusercontent.com/archettialberto/tilebased_navigation_datasets/{UPSTREAM_REV}/{key}.npz'
            if path.exists():
                if sha256(path) != expected:
                    raise ValueError(f'{path}: checksum mismatch; move the corrupt file before retrying')
                print(f'Verified existing {path}', flush=True)
            else:
                if offline:
                    raise FileNotFoundError(f'{path}: missing in offline mode')
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix('.npz.part')
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as out:
                        shutil.copyfileobj(response, out)
                    if sha256(partial) != expected:
                        raise ValueError(f'{path}: download checksum mismatch (possibly an HTML error/LFS pointer)')
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
                print(f'Downloaded {path}', flush=True)
            records.append({'path': key + '.npz', 'sha256': expected, 'url': url})
    manifest = Path(root) / 'download_manifest.json'
    manifest.write_text(json.dumps({'upstream_revision': UPSTREAM_REV, 'files': records}, indent=2) + '\n')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--dataset', choices=['pkmn', 'warcraft', 'all'], default='pkmn')
    p.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    p.add_argument('--offline', action='store_true', help='only verify files already copied to output-root')
    a = p.parse_args()
    download(a.output_root, ['pkmn', 'warcraft'] if a.dataset == 'all' else [a.dataset], a.splits, a.offline)

if __name__ == '__main__':
    main()
