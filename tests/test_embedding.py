"""Dependency-free correctness checks; fake vectors do not measure semantics."""
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding import EmbeddingConfig, TextRecord, Vectorizer, VectorStore, load_records
from embedding.encoder import ApiBackend, LocalBackend
from embedding.store import inspect_store
from embedding.build_vectors import main


class FakeBackend:
    def __init__(self):
        self.seen = []

    def encode(self, texts):
        self.seen.extend(texts)
        return [[len(text), 1, 2] for text in texts]


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = EmbeddingConfig('local', 'test-only', 'v1', chunk_chars=4)
        self.backend = FakeBackend()
        self.encoder = Vectorizer(self.config, batch_size=2, backend=self.backend)
        self.path = self.root / 'vectors.sqlite'

    def record(self, identity='1', text='患者：失眠三天'):
        return TextRecord(identity, text, 'dialog', 'test-fixture')

    def test_dialog_preserves_roles_and_excludes_labels(self):
        path = self.root / 'data.txt'
        path.write_text(json.dumps({'dialog': ['患者：症状', '医生：问题'],
                                    'label': ['PRIVATE_DRUG'], 'disease': ['PRIVATE_DISEASE']}))
        records = load_records(path, 'dialog')
        self.assertEqual(records[0].text, '患者：症状\n医生：问题')
        self.encoder.encode_texts([records[0].text])
        self.assertNotIn('PRIVATE', ''.join(self.backend.seen))

    def test_memory_ids_and_empty_files(self):
        path = self.root / 'memory.json'
        path.write_text('[]')
        self.assertEqual(load_records(path, 'reflection'), [])
        path.write_text(json.dumps([{'case_id': 8, 'reflection': '核对否定事实'}]))
        self.assertEqual(load_records(path, 'reflection')[0].record_id, '8')
        path.write_text(json.dumps([{'memory_id': 'stable', 'experience': '保留不确定性'}]))
        self.assertEqual(load_records(path, 'experience')[0].record_id, 'stable')

    def test_chunks_cover_full_text_normalized_and_batched(self):
        result = self.encoder.encode_texts(['abcdefghij', '中文'])
        self.assertEqual(self.backend.seen, ['abcd', 'efgh', 'ij', '中文'])
        self.assertEqual(self.encoder.last_stats['backend_calls'], 2)
        for vector in result:
            self.assertAlmostEqual(sum(x*x for x in vector), 1)
        self.assertEqual(self.encoder.encode_texts([]), [])
        with self.assertRaises(ValueError):
            self.encoder.encode_texts(['  '])

    def test_initial_repeat_add_edit_delete_and_namespace_isolation(self):
        with VectorStore(self.path, self.config) as store:
            self.assertEqual(store.sync('train', [self.record()], self.encoder)['encoded_texts'], 1)
            self.backend.seen.clear()
            repeat = store.sync('train', [self.record()], self.encoder)
            self.assertEqual(repeat['encoded_texts'], 0)
            self.assertEqual(self.backend.seen, [])
            added = store.sync('train', [self.record(), self.record('2', '新对话')], self.encoder)
            self.assertEqual(added['encoded_texts'], 1)
            changed = store.sync('train', [self.record(text='修改对话')], self.encoder)
            self.assertEqual(changed['updated'], 1)
            self.assertEqual(changed['active_records'], 2)
            store.sync('test', [self.record('2', '新对话')], self.encoder)
            removed = store.sync('train', [], self.encoder, prune=True)
            self.assertEqual(removed['deleted'], 2)
            self.assertEqual(store.get('test', '2')['text'], '新对话')
            with self.assertRaises(KeyError):
                store.get('train', '2')

    def test_identical_content_encoded_once_and_float32_roundtrip(self):
        with VectorStore(self.path, self.config) as store:
            stats = store.sync('train', [self.record('1'), self.record('2')], self.encoder)
            self.assertEqual(stats['encoded_texts'], 1)
            self.assertEqual(stats['new_vector_bytes'], 12)
            expected = store.get('train', '1')['vector']
        with VectorStore(self.path, self.config) as store:
            self.assertEqual(store.get('train', '2')['vector'], expected)
            self.assertAlmostEqual(sum(x*x for x in expected), 1, places=6)
        info = inspect_store(self.path)
        self.assertEqual(info['namespaces'], {'train': 2})
        self.assertEqual(info['cached_vectors'], 1)

    def test_configuration_mismatch_rejected(self):
        with VectorStore(self.path, self.config):
            pass
        for change in ({'model_version': 'v2'}, {'chunk_chars': 8}, {'prefix': 'query: '}):
            config = EmbeddingConfig(**{**self.config.__dict__, **change})
            with self.assertRaisesRegex(ValueError, '配置不同'):
                VectorStore(self.path, config)

    def test_failed_import_rolls_back_records_and_vectors(self):
        with VectorStore(self.path, self.config) as store:
            store.sync('train', [self.record()], self.encoder)
            before = inspect_store(self.path)
            with patch.object(self.backend, 'encode', return_value=[[1, 2]]):
                with self.assertRaises(ValueError):
                    store.sync('train', [self.record('new', '新')], self.encoder, prune=True)
            self.assertEqual(store.get('train', '1')['text'], self.record().text)
            self.assertEqual(inspect_store(self.path), before)

    def test_invalid_vectors_rejected(self):
        for vector in ([0, 0], [math.nan, 1], [math.inf], [True], [], ['1']):
            with self.subTest(vector=vector):
                with patch.object(self.backend, 'encode', return_value=[vector]):
                    with self.assertRaises(ValueError):
                        self.encoder.encode_texts(['x'])

    def test_duplicate_ids_rejected_before_encoding(self):
        with VectorStore(self.path, self.config) as store:
            with self.assertRaises(ValueError):
                store.sync('train', [self.record(), self.record()], self.encoder)
        self.assertEqual(self.backend.seen, [])

    def test_api_order_validation_and_secret_not_in_manifest(self):
        config = EmbeddingConfig('api', 'example-model', 'v1', endpoint='https://example.invalid/embeddings')
        backend = ApiBackend(config, 'TEST_EMBEDDING_KEY', 5)
        with patch.dict('os.environ', {'TEST_EMBEDDING_KEY': 'secret'}):
            response = {'data': [{'index': 1, 'embedding': [0, 1]},
                                 {'index': 0, 'embedding': [1, 0]}]}
            with patch('embedding.encoder.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as opened:
                self.assertEqual(backend.encode(['a', 'b']), [[1, 0], [0, 1]])
                body = json.loads(opened.call_args.args[0].data)
                self.assertEqual(body['input'], ['a', 'b'])
            response['data'][1]['index'] = 1
            with patch('embedding.encoder.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaises(ValueError):
                    backend.encode(['a', 'b'])
        self.assertNotIn('secret', json.dumps(config.manifest))

    def test_local_backend_rejects_silent_truncation(self):
        backend = LocalBackend(self.config)
        class Model:
            max_seq_length = 2
            def tokenizer(self, *args, **kwargs):
                return {'input_ids': [[1, 2, 3]]}
        backend.model = Model()
        with self.assertRaisesRegex(ValueError, 'token 上限'):
            backend.encode(['too long'])

    def test_cli_build_inspect_and_repeat(self):
        source = self.root / 'input.jsonl'
        source.write_text(json.dumps({'id': 'a', 'text': '示例对话'}))
        args = ['build', '--source', str(source), '--kind', 'text', '--namespace', 'demo',
                '--output', str(self.path), '--backend', 'local', '--model', 'test-only',
                '--model-version', 'v1']
        with patch('embedding.encoder.LocalBackend.encode', side_effect=self.backend.encode):
            for expected in (1, 0):
                with patch('sys.stdout', new_callable=io.StringIO) as stdout:
                    main(args)
                    self.assertEqual(json.loads(stdout.getvalue())['encoded_texts'], expected)
        with patch('sys.stdout', new_callable=io.StringIO) as stdout:
            main(['inspect', '--output', str(self.path)])
            self.assertEqual(json.loads(stdout.getvalue())['namespaces'], {'demo': 1})
        with patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                main(args + ['--limit', '1', '--prune'])


if __name__ == '__main__':
    unittest.main()
