import json
from pathlib import Path
import tempfile
import unittest
from episode_manifest import load_manifest


class ManifestTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';p.write_text(json.dumps({'episodes':rows}))
            return load_manifest(p)

    def test_explicit_split(self):
        rows=self.load([{'seed':8,'split':'validation','path':'/tmp/seed8'},
                        {'seed':0,'split':'train','path':'/tmp/seed0'}])
        self.assertEqual([r['seed'] for r in rows],[0,8])

    def test_no_seed_leakage(self):
        with self.assertRaises(ValueError):
            self.load([{'seed':8,'split':'validation','path':'/tmp/seed8'},
                       {'seed':8,'split':'train','path':'/tmp/other'}])

    def test_no_path_leakage(self):
        with self.assertRaises(ValueError):
            self.load([{'seed':8,'split':'validation','path':'/tmp/same'},
                       {'seed':0,'split':'train','path':'/tmp/same'}])

    def test_requires_both_splits(self):
        with self.assertRaises(ValueError):
            self.load([{'seed':0,'split':'train','path':'/tmp/seed0'}])


if __name__=='__main__':unittest.main()
