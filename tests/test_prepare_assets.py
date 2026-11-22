import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.prepare_assets import prepare


class PrepareAssetsTests(unittest.TestCase):
    def fixture(self, root):
        raw = b'{"n":1}\r\n{"n":2}\r\n'
        subset = b'{"n":2}\r\n{"n":1}\r\n'
        (root / 'full.txt').write_bytes(raw)
        sha = lambda b: hashlib.sha256(b).hexdigest()
        manifest = {'subsets': {'subset.txt': {'source': 'full.txt',
                    'source_sha256': sha(raw), 'lines': [2, 1], 'sha256': sha(subset)}},
                    'copies': {'copy.txt': {'source': 'subset.txt', 'sha256': sha(subset)}}}
        (root / 'assets_manifest.json').write_text(json.dumps(manifest))
        return subset, manifest

    def test_exact_order_and_crlf_preserved_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected, _ = self.fixture(root)
            prepare(root)
            self.assertEqual((root / 'subset.txt').read_bytes(), expected)
            self.assertEqual((root / 'copy.txt').read_bytes(), expected)
            prepare(root)

    def test_modified_output_prevents_any_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / 'copy.txt').write_bytes(b'new experiment')
            with self.assertRaisesRegex(ValueError, '已修改'):
                prepare(root)
            self.assertFalse((root / 'subset.txt').exists())
            self.assertEqual((root / 'copy.txt').read_bytes(), b'new experiment')

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / 'full.txt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '源数据'):
                prepare(root)

    def test_manifest_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, manifest = self.fixture(root)
            manifest['copies']['../escape.txt'] = manifest['copies'].pop('copy.txt')
            (root / 'assets_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, '路径越界'):
                prepare(root)
            self.assertFalse((root / 'subset.txt').exists())

class CompressedAssetsTests(unittest.TestCase):
    def test_gzip_source_restores_original_and_dependent_subset(self):
        import gzip
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b'{"id":1}\n{"id":2}\n'
            packed = gzip.compress(raw, mtime=0)
            (root / 'full.txt.gz').write_bytes(packed)
            sha = lambda b: hashlib.sha256(b).hexdigest()
            manifest = {'compressed': {'full.txt': {'source': 'full.txt.gz',
                        'source_sha256': sha(packed), 'sha256': sha(raw)}},
                        'subsets': {'subset.txt': {'source': 'full.txt',
                        'source_sha256': sha(raw), 'lines': [2], 'sha256': sha(b'{"id":2}\n')}},
                        'copies': {}}
            (root / 'assets_manifest.json').write_text(json.dumps(manifest))
            prepare(root)
            self.assertEqual((root / 'full.txt').read_bytes(), raw)
            self.assertEqual((root / 'subset.txt').read_bytes(), b'{"id":2}\n')

    def test_gzip_wrong_hash_writes_nothing(self):
        import gzip
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'data.gz').write_bytes(gzip.compress(b'input', mtime=0))
            manifest = {'compressed': {'data.txt': {'source': 'data.gz',
                        'source_sha256': 'wrong', 'sha256': 'wrong'}}, 'subsets': {}, 'copies': {}}
            (root / 'assets_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, '压缩源文件'):
                prepare(root)
            self.assertFalse((root / 'data.txt').exists())
