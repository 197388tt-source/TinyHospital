"""Semantic routing tests with a local embeddings API; no external models or keys."""
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from cases import DialMedCase
from embedding import EmbeddingConfig, TextRecord, Vectorizer, VectorStore
from embedding.store import inspect_store
from history.memory import CaseMemoryStore
from hospital_process import ProcessHospitalSession
from main import HospitalSession
from performance_baseline import FixedModel


class InvalidDoctorModel(FixedModel):
    def chat(self, messages):
        if messages[0]['content'].startswith(('你是一名专业的医生', '你是一名医生')):
            return '{}'
        return super().chat(messages)


class EmbeddingsHandler(BaseHTTPRequestHandler):
    inputs = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).inputs.extend(request['input'])
        data = []
        for i, text in enumerate(request['input']):
            # Deliberately recognize synonyms with no shared lexical features.
            vector = [1.0, 0.0] if any(word in text for word in ('insomnia', 'sleepless')) else [0.0, 1.0]
            data.append({'index': i, 'embedding': vector})
        raw = json.dumps({'data': data}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class SemanticRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), EmbeddingsHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.golden, self.reflection = self.root/'golden.json', self.root/'reflection.json'
        self.dataset = self.root/'train.txt'
        self.dataset.write_text(''.join(json.dumps({'dialog': [text], 'label': ['酮康唑'],
                                                    'disease': 'PRIVATE_LABEL'}) + '\n'
                                        for text in ('insomnia', 'rash', 'sleepless')), encoding='utf-8')
        self.write(self.golden, [])
        self.write(self.reflection, [])
        self.config = EmbeddingConfig('api', 'test-semantic', 'v1', chunk_chars=256,
                                     endpoint=f'http://127.0.0.1:{self.server.server_port}/embeddings')
        self.options = {'config': self.config, 'path': self.root/'vectors.sqlite',
                        'api_key_env': 'SEMANTIC_TEST_KEY'}
        key = patch.dict(os.environ, {'SEMANTIC_TEST_KEY': 'local-test-only'})
        key.start()
        self.addCleanup(key.stop)
        EmbeddingsHandler.inputs = []

    @staticmethod
    def write(path, rows):
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')

    def memory(self, training=True, mode='embedding'):
        return CaseMemoryStore(self.golden, self.reflection, training, self.dataset,
                               retrieval_mode=mode, embedding_options=self.options)

    def test_empty_then_save_encodes_and_semantic_hit_beats_jaccard(self):
        memory = self.memory()
        self.assertEqual(memory.retrieve('sleepless', 3, str(self.dataset))['golden_experiences'], [])
        self.assertEqual(EmbeddingsHandler.inputs, [])
        memory.save_success(1, '应核对睡眠问题的病程。')
        self.assertEqual(inspect_store(self.options['path'])['cached_vectors'], 1)
        with VectorStore(self.options['path'], self.config) as store:
            indexed = store.get(memory.semantic.namespace, 'golden_experiences:1')
        self.assertEqual(indexed['text'], 'insomnia')
        self.assertEqual(indexed['kind'], 'dialog')
        result = memory.retrieve('sleepless', 3, str(self.dataset))
        self.assertEqual(result['golden_experiences'][0]['case_id'], 1)
        semantic_stats = memory.retrieval_statistics['embedding']
        self.assertEqual(semantic_stats['search_backend'], 'numpy_exact')
        self.assertEqual(semantic_stats['matrix_rows'], 1)
        self.assertEqual(semantic_stats['matrix_dimension'], 2)
        self.assertEqual(semantic_stats['matrix_bytes'], 8)
        self.assertEqual(semantic_stats['searches'], 1)
        self.assertEqual(self.memory(mode='jaccard').retrieve('sleepless', 3, str(self.dataset))['golden_experiences'], [])
        self.assertIn('insomnia', EmbeddingsHandler.inputs)
        self.assertFalse(any('应核对睡眠问题' in text for text in EmbeddingsHandler.inputs))
        self.assertTrue(all('PRIVATE_LABEL' not in text and '酮康唑' not in text for text in EmbeddingsHandler.inputs))
        calls = len(EmbeddingsHandler.inputs)
        memory.retrieve('sleepless', 3, str(self.dataset))
        self.assertEqual(len(EmbeddingsHandler.inputs), calls)
        self.assertEqual(memory.retrieval_statistics['embedding']['query_cache_hits'], 1)

    def test_external_edits_deletes_and_frozen_snapshot(self):
        self.write(self.golden, [{'case_id': 1, 'experience': '旧经验'}])
        train, frozen = self.memory(), self.memory(False)
        frozen.retrieve('sleepless', 3, str(self.dataset))
        self.write(self.golden, [{'case_id': 1, 'experience': '新经验'}])
        self.assertEqual(train.retrieve('sleepless', 3, str(self.dataset))['golden_experiences'][0]['experience'], '新经验')
        encoded_inputs = len(EmbeddingsHandler.inputs)
        self.write(self.golden, [{'case_id': 1, 'experience': '再次更新的经验'}])
        self.assertEqual(
            train.retrieve('sleepless', 3, str(self.dataset))['golden_experiences'][0]['experience'],
            '再次更新的经验',
        )
        self.assertEqual(len(EmbeddingsHandler.inputs), encoded_inputs)
        self.assertEqual(frozen.retrieve('sleepless', 3, str(self.dataset))['golden_experiences'][0]['experience'], '旧经验')
        self.write(self.golden, [])
        self.assertEqual(train.retrieve('sleepless', 3, str(self.dataset))['golden_experiences'], [])
        self.assertEqual(sum(inspect_store(self.options['path'])['namespaces'].values()), 0)
        self.assertEqual(frozen.retrieve('sleepless', 3, str(self.dataset))['golden_experiences'][0]['experience'], '旧经验')

    def test_replacement_self_exclusion_and_cache_restart(self):
        memory = self.memory()
        memory.save_success(1, '旧经验')
        memory.save_reflection(1, '新反思')
        result = memory.retrieve('sleepless', 3, str(self.dataset))
        self.assertFalse(result['golden_experiences'])
        self.assertEqual(result['reflections'][0]['reflection'], '新反思')
        self.assertFalse(memory.retrieve('sleepless', 1, str(self.dataset))['reflections'])
        self.assertFalse(memory.retrieve('insomnia', 3, str(self.dataset))['reflections'])
        other = self.memory()
        other.retrieve('sleepless', 3, str(self.dataset))
        self.assertEqual(other.retrieval_statistics['embedding']['encoded_texts'], 0)
        self.assertFalse(self.memory(mode='none').retrieve('sleepless', 3, str(self.dataset))['reflections'])

    def test_vector_only_excludes_identical_source_dialog(self):
        from hashlib import sha256
        memory = self.memory()
        memory.save_success(1, '不得检索同一对话')
        result = memory.retrieve(None, 99, 'other-test.txt', query_vector=[1.0, 0.0],
                                 query_text_hash=sha256(b'insomnia').hexdigest())
        self.assertEqual(result['golden_experiences'], [])

    def test_api_failure_is_explicit_and_retry_recovers(self):
        memory = self.memory()
        with patch.dict(os.environ, {'SEMANTIC_TEST_KEY': ''}):
            with self.assertRaisesRegex(ValueError, 'SEMANTIC_TEST_KEY'):
                memory.save_success(1, '待重试经验')
        self.assertEqual(memory.retrieve('sleepless', 3, str(self.dataset))['golden_experiences'][0]['experience'], '待重试经验')

    def test_store_top_k_dimension_namespace_and_exclusion(self):
        encoder = Vectorizer(self.config, api_key_env='SEMANTIC_TEST_KEY')
        with VectorStore(self.options['path'], self.config) as store:
            store.sync('a', [TextRecord('1', 'insomnia', 'text', 'test'),
                             TextRecord('2', 'rash', 'text', 'test')], encoder)
            self.assertEqual(store.search('a', [1, 0], limit=1)[0]['record_id'], '1')
            self.assertEqual(store.search('a', [1, 0], min_similarity=0.5, exclude_ids=['1']), [])
            self.assertEqual(store.search('b', [1, 0]), [])
            with self.assertRaisesRegex(ValueError, '维度'):
                store.search('a', [1, 0, 0])

    def session_paths(self):
        return dict(verbose=False, golden_path=self.golden, reflection_path=self.reflection,
                    dataset_path=self.dataset, communication_path=self.root/'traffic.jsonl',
                    control_path=self.root/'control.jsonl', llm_path=self.root/'llm.jsonl',
                    embedding_options=self.options)

    def test_sync_default_embedding_reaches_actual_doctor_prompt(self):
        self.write(self.golden, [{'case_id': 1, 'experience': '睡眠经验进入医生提示词'}])
        model = FixedModel()
        with patch('main.LLMClient', return_value=model):
            session = HospitalSession(training=False, **self.session_paths())
            try:
                session.run_case(DialMedCase(3, json.dumps(['sleepless']), ('酮康唑',)))
                self.assertIn('睡眠经验进入医生提示词', model.prompts[0][1]['content'])
                self.assertEqual(session.memories.retrieval_statistics['retrieval_mode'], 'embedding')
            finally:
                session.close()

    def test_training_saves_memory_without_retrieval_then_test_uses_it(self):
        # Training mines experience without retrieval; test loads the frozen memory.
        with ProcessHospitalSession(training=True, llm_client_factory=FixedModel,
                                    **self.session_paths()) as session:
            session.run_case(DialMedCase(1, json.dumps(['insomnia']), ('依巴斯汀',)))
            session.run_case(DialMedCase(3, json.dumps(['sleepless']), ('酮康唑',)))
            self.assertFalse(Path(self.options['path']).exists())
            self.assertEqual(session.retrieval_statistics['retrieval_mode'], 'none')
            self.assertEqual(session.retrieval_statistics['queries'], 0)
        self.assertTrue(json.loads(self.reflection.read_text()))
        with ProcessHospitalSession(training=False, llm_client_factory=FixedModel,
                                    **self.session_paths()) as session:
            session.run_case(DialMedCase(3, json.dumps(['sleepless']), ('酮康唑',)))
            self.assertGreater(inspect_store(self.options['path'])['cached_vectors'], 0)
            self.assertGreater(session.retrieval_statistics['hits'], 0)

    def test_vector_transports_feed_same_doctor_input_without_reencoding(self):
        from shared_transport import attach_shared_memory
        self.write(self.golden, [{'case_id': 1, 'experience': '睡眠经验进入医生提示词'}])
        inputs = []
        for mode in ('text', 'binary', 'shared_memory'):
            with self.subTest(mode=mode):
                with ProcessHospitalSession(training=False, llm_client_factory=FixedModel,
                                            vector_transport=mode, **self.session_paths()) as session:
                    session.run_case(DialMedCase(3, json.dumps(['sleepless']), ('酮康唑',)))
                    packets = [p for p in session.trace.packets
                               if p.payload.get('capability') == 'doctor.recommend'
                               and p.head.source == 'RetrievalAgent']
                    inputs.append(json.loads(packets[-1].payload['text']))
                    stats = session.retrieval_statistics
                    if mode != 'text':
                        self.assertEqual(stats['query_sender']['query_encodings'], 1)
                        self.assertEqual(stats['embedding']['query_encodings'], 0)
                        self.assertEqual(stats['vector_receive']['received_vectors'], 1)
                        requests = [p for p in session.trace.packets
                                    if p.payload.get('capability') == 'retrieval.history'
                                    and p.head.source == 'PatientAgent']
                        request_body = json.loads(requests[-1].payload['text'])
                        self.assertNotIn('patient_dialog', request_body)
                        self.assertIn('query_embedding', request_body)
                    else:
                        requests = [p for p in session.trace.packets
                                    if p.payload.get('capability') == 'retrieval.history'
                                    and p.head.source == 'PatientAgent']
                        request_body = json.loads(requests[-1].payload['text'])
                        self.assertIn('patient_dialog', request_body)
                        self.assertNotIn('query_embedding', request_body)
                    if mode == 'binary':
                        frames = [f for f in session.runtime.binary_frame_stats
                                  if f['capability'] in ('patient.request_history', 'retrieval.history')]
                        self.assertEqual(len(frames), 2)
                        self.assertEqual(sum(f['binary_bytes'] for f in frames), 16)
                    if mode == 'shared_memory':
                        self.assertEqual(stats['vector_receive']['shared_read_bytes'], 8)
                        requests = [p for p in session.trace.packets
                                    if p.payload.get('capability') == 'retrieval.history'
                                    and p.head.source == 'PatientAgent']
                        ref = json.loads(requests[-1].payload['text'])['query_embedding']['reference']
                        with self.assertRaises(FileNotFoundError):
                            attach_shared_memory(ref['name'])
        self.assertEqual(inputs[0], inputs[1])
        self.assertEqual(inputs[1], inputs[2])
        self.assertNotIn('query_embedding', inputs[0])

    def test_shared_vectors_compose_with_incremental_and_structured_transport(self):
        self.write(self.golden, [{'case_id': 1, 'experience': '睡眠经验进入医生提示词'}])
        with ProcessHospitalSession(training=False, llm_client_factory=FixedModel,
                                    vector_transport='shared_memory', incremental_history=True,
                                    shared_dialog=True, structured_data=True,
                                    **self.session_paths()) as session:
            session.run_case(DialMedCase(3, json.dumps(['sleepless']), ('酮康唑',)))
            packets = [p for p in session.trace.packets
                       if p.payload.get('capability') == 'doctor.recommend'
                       and p.head.source == 'RetrievalAgent']
            body = packets[-1].payload['data']
            self.assertNotIn('query_embedding', body)
            self.assertEqual(body['history']['golden_experiences'], ['睡眠经验进入医生提示词'])
            self.assertEqual(session.retrieval_statistics['vector_receive']['received_vectors'], 1)

    def test_vector_descriptor_rejects_corruption_and_wrong_binding(self):
        from copy import deepcopy
        from embedding.transport import QueryVectorSender, receive_vector
        sender = QueryVectorSender('binary', self.options)
        descriptor, raw = sender.prepare('sleepless', 17)
        args = dict(trace_id=17, fingerprint=self.config.fingerprint, dialog='sleepless')
        self.assertEqual(receive_vector(descriptor, binary=raw, **args), [1.0, 0.0])
        with self.assertRaises(ValueError):
            receive_vector(descriptor, binary=raw[:-1], **args)
        for field, value in (('trace_id', 18), ('dimension', 3),
                             ('model_fingerprint', '0'*64), ('text_hash', '0'*64)):
            invalid = deepcopy(descriptor)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                receive_vector(invalid, binary=raw, **args)

    def test_shared_vector_released_when_doctor_fails(self):
        from shared_transport import attach_shared_memory
        with ProcessHospitalSession(training=False, llm_client_factory=InvalidDoctorModel,
                                    vector_transport='shared_memory', **self.session_paths()) as session:
            with self.assertRaises(RuntimeError):
                session.run_case(DialMedCase(3, json.dumps(['sleepless']), ('酮康唑',)))
            packets = [p for p in session.trace.packets
                       if p.payload.get('capability') == 'retrieval.history'
                       and p.head.source == 'PatientAgent']
            ref = json.loads(packets[-1].payload['text'])['query_embedding']['reference']
            with self.assertRaises(FileNotFoundError):
                attach_shared_memory(ref['name'])


if __name__ == '__main__':
    unittest.main()
