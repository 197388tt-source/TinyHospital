import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases import (DIALMED_DEV_PATH, DIALMED_TEST_PATH, DIALMED_TRAIN_PATH, default_dataset_path,
                   memory_source_dataset_path)
from main import DialMedCase, HospitalSession, State, load_dialmed_cases
from tool.llm_client import TokenUsage


class FakeLLMClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.last_usage = TokenUsage(10, 5, 15)

    def chat(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def recommendation(drug, reason=None):
    reason = reason or ('关键证据：头皮屑伴瘙痒；疾病判断：考虑头皮屑相关问题，仍需确认；'
                        '选药依据：酮康唑适应症覆盖头皮屑和瘙痒。')
    return json.dumps({'recommended_drugs': [drug], 'reasoning_summary': reason},
                      ensure_ascii=False)


def supported_golden():
    return json.dumps({
        'supported': True, 'evidence': [{'drug': '酮康唑',
        'case_evidence': '头皮屑伴瘙痒',
        'drug_evidence': '头皮屑及其相关的脱屑、鳞屑和瘙痒'}],
        'experience': ('当头皮屑伴瘙痒时，应核对适应症并选择针对头皮屑相关症状的药物，'
                       '因为其适应症覆盖脱屑和瘙痒。'),
    }, ensure_ascii=False)


def unsupported_golden():
    return json.dumps({'supported': False, 'evidence': [], 'experience': ''}, ensure_ascii=False)


class TinyHospitalFlowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.golden = root / 'golden_experience.json'
        self.reflections = root / 'reflection_memory.json'
        self.traffic = root / 'token_usage.jsonl'
        self.control = root / 'control_communication.jsonl'
        self.llm = root / 'llm_usage.jsonl'
        self.reports = root / 'reports.json'
        self.history = root / 'history_memory.json'
        self.dataset = root / 'train.txt'
        self.dialog = '患者：头皮屑伴瘙痒三天'
        self.dataset.write_text(json.dumps({
            'dialog': [self.dialog], 'label': ['酮康唑']
        }, ensure_ascii=False) + '\n', encoding='utf-8')

    def session(self, training=True):
        return HospitalSession(
            retrieval_mode='jaccard', verbose=False, golden_path=self.golden, reflection_path=self.reflections,
            communication_path=self.traffic, control_path=self.control,
            llm_path=self.llm, dataset_path=self.dataset, training=training,
        )

    def test_dataset_dialog_is_unchanged(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 3)
        self.assertEqual([case.case_id for case in cases], [1, 2, 3])
        with DIALMED_TRAIN_PATH.open(encoding='utf-8') as source:
            for case, line in zip(cases, source):
                self.assertEqual(json.loads(case.patient_input), json.loads(line)['dialog'])

    def test_cut_datasets_are_defaults_and_test_memories_map_to_train_cut(self):
        self.assertEqual(DIALMED_TRAIN_PATH.name, 'traincut.txt')
        self.assertEqual(DIALMED_TEST_PATH.name, 'testcut.txt')
        self.assertEqual(default_dataset_path(True), DIALMED_TRAIN_PATH)
        self.assertEqual(default_dataset_path(False), DIALMED_TEST_PATH)
        self.assertEqual(memory_source_dataset_path(DIALMED_TEST_PATH, False),
                         DIALMED_TRAIN_PATH)
        self.assertEqual(memory_source_dataset_path(DIALMED_TRAIN_PATH, True),
                         DIALMED_TRAIN_PATH)

    def test_validation_memories_map_to_training_source(self):
        for path in (DIALMED_DEV_PATH, DIALMED_DEV_PATH.parent / '..' / 'data' / 'devcut.txt'):
            self.assertEqual(memory_source_dataset_path(path, False), DIALMED_TRAIN_PATH)
        self.assertEqual(memory_source_dataset_path(DIALMED_DEV_PATH, True), DIALMED_DEV_PATH)
        custom = Path('/tmp/custom-data/devcut.txt')
        self.assertEqual(memory_source_dataset_path(custom, False), custom)

    def test_training_skips_history_even_with_existing_memories(self):
        self.golden.write_text(json.dumps([{'case_id': 1, 'experience': '已有经验不应进入训练'}]))
        fake = FakeLLMClient([recommendation('酮康唑'), supported_golden(),
                              recommendation('酮康唑'), supported_golden()])
        with patch('main.LLMClient', return_value=fake), \
                patch('history.memory.CaseMemoryStore.retrieve',
                      side_effect=AssertionError('训练不能查询经验')), \
                patch('embedding.encoder.ApiBackend.encode',
                      side_effect=AssertionError('训练不能调用嵌入接口')):
            session = HospitalSession(verbose=False, training=True,
                                      golden_path=self.golden, reflection_path=self.reflections,
                                      dataset_path=self.dataset, llm_path=self.llm,
                                      communication_path=self.traffic, control_path=self.control)
            try:
                for case_id in (1, 2):
                    session.run_case(DialMedCase(case_id, self.dialog, ('酮康唑',)))
                self.assertIsNone(session.patient.vector_sender)
                self.assertIsNone(session.memories.semantic)
                self.assertEqual(session.retrieval.history_query_count, 0)
                self.assertEqual(session.retrieval_statistics['queries'], 0)
                packets = [p for p in session.trace.packets
                           if p.payload.get('capability') == 'doctor.recommend']
                self.assertEqual(len(packets), 2)
                for packet in packets:
                    self.assertEqual(packet.head.source, 'PatientAgent')
                    body = json.loads(packet.payload['text'])
                    for key in ('history', 'query_embedding', 'actual_drugs', 'drug_labels'):
                        self.assertNotIn(key, body)
            finally:
                session.close()
        self.assertEqual(len(json.loads(self.golden.read_text())), 2)

    def test_success_saves_only_approved_experience_and_logs_its_tokens(self):
        fake = FakeLLMClient([recommendation('酮康唑'), supported_golden()])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
            self.assertTrue(outcome['exact_match'])
            self.assertIsNotNone(outcome['golden_experience'])
            self.assertEqual(len(fake.calls), 2)
            self.assertIn('医生判断依据', fake.calls[1][1]['content'])
            self.assertIn('推荐药物说明', fake.calls[1][1]['content'])
            self.assertNotIn('真实药物：', fake.calls[0][1]['content'])
            self.assertFalse(self.history.exists())
            self.assertFalse(self.reports.exists())
            self.assertFalse(self.reflections.exists())
            self.assertEqual(json.loads(self.golden.read_text(encoding='utf-8')),
                             [{'case_id': 1, 'experience': outcome['golden_experience']}])
            rows = [json.loads(row) for row in self.llm.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([row['record_type'] for row in rows],
                             ['llm_call', 'llm_call', 'case_total', 'golden_experience_total'])
            self.assertEqual(rows[-1]['total_tokens'], 15)
            self.assertTrue(all(agent.state == State.IDLE for agent in session.agents))
            self.assertNotIn('SummarizerAgent', session.runtime.agents)
            session.close()

    def test_unsupported_success_writes_no_correct_case_or_experience(self):
        fake = FakeLLMClient([recommendation('酮康唑'), unsupported_golden()])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
            self.assertTrue(outcome['exact_match'])
            self.assertIsNone(outcome['golden_experience'])
            self.assertFalse(self.history.exists())
            self.assertFalse(self.golden.exists())
            self.assertFalse(self.reports.exists())
            session.close()

    def test_unsubstantiated_positive_lesson_is_rejected(self):
        invalid = json.loads(supported_golden())
        invalid['evidence'][0]['case_evidence'] = '患者已确诊真菌感染'
        fake = FakeLLMClient([recommendation('酮康唑'),
                              *[json.dumps(invalid, ensure_ascii=False)] * 3])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
            self.assertTrue(outcome['exact_match'])
            self.assertIsNone(outcome['golden_experience'])
            self.assertFalse(self.golden.exists())
            session.close()

    def test_golden_validation_reports_specific_evidence_errors(self):
        from agent.validation import GoldenValidationError, _validate_golden_experience
        base = json.loads(supported_golden())
        cases = [
            ('case_evidence', '', 'case_evidence_empty'),
            ('case_evidence', '痒', 'case_evidence_short'),
            ('case_evidence', '[MASK]', 'case_evidence_mask'),
            ('case_evidence', '不是原文', 'case_evidence_not_verbatim'),
            ('drug', '不在推荐列表', 'drug_not_recommended'),
            ('drug_evidence', '', 'drug_evidence_empty'),
            ('drug_evidence', '痒', 'drug_evidence_short'),
            ('drug_evidence', '不是原文', 'drug_evidence_not_verbatim'),
            ('experience', '字' * 201, 'experience_too_long'),
        ]
        details = {'酮康唑': {'适应症': base['evidence'][0]['drug_evidence']}}
        for field, value, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(GoldenValidationError) as caught:
                    _validate_golden_experience(
                        ({**base, field: value} if field == 'experience' else
                         {**base, 'evidence': [{**base['evidence'][0], field: value}]}),
                        dialog=self.dialog,
                        reasoning_summary='头皮屑和瘙痒', drugs=['酮康唑'], introductions=details)
                self.assertEqual(caught.exception.code, code)
        with self.assertRaises(GoldenValidationError) as caught:
            _validate_golden_experience(base, dialog=self.dialog,
                reasoning_summary='头皮屑和瘙痒', drugs=['酮康唑'], introductions={})
        self.assertEqual(caught.exception.code, 'indication_missing')

    def test_golden_validation_does_not_audit_semantics(self):
        from agent.validation import _validate_golden_experience
        # Valid sources need not share words with each other or the lesson.
        # Whether they support the lesson is the expert model's responsibility.
        answer = {
            'supported': True,
            'evidence': [{'drug': '药物甲', 'case_evidence': '发热、咳嗽',
                          'drug_evidence': '头皮屑反复'}],
            'experience': '',
        }
        for lesson in ('匹配不代表确诊，也不保证绝对安全。',
                       '由专业人员结合完整材料作进一步判断。', '药物甲'):
            with self.subTest(lesson=lesson):
                answer['experience'] = lesson
                self.assertEqual(_validate_golden_experience(
                    answer, dialog='患者：发热、咳嗽', reasoning_summary='尚待评估',
                    drugs=['药物甲'], introductions={'药物甲': {'适应症': '头皮屑反复'}},
                ), lesson)

    def test_golden_evidence_array_contract(self):
        from agent.validation import GoldenValidationError, _validate_golden_experience
        base = json.loads(supported_golden())
        entry = base['evidence'][0]
        second = {**entry, 'drug': '另一推荐药'}
        details = {drug: {'适应症': entry['drug_evidence']}
                   for drug in ['酮康唑', '另一推荐药']}

        def validate(answer, **overrides):
            args = dict(dialog=json.dumps([self.dialog], ensure_ascii=False),
                        reasoning_summary='头皮屑和瘙痒', drugs=list(details),
                        introductions=details)
            args.update(overrides)
            return _validate_golden_experience(answer, **args)

        # A lesson may cite one or several of the recommended medicines.
        self.assertEqual(validate(base), base['experience'])
        self.assertEqual(validate({**base, 'evidence': [entry, second]}), base['experience'])
        self.assertIsNone(validate(json.loads(unsupported_golden())))
        invalid_cases = [
            ({**base, 'drug': '酮康唑'}, 'response_fields'),
            ({**base, 'supported': 'true'}, 'supported_type'),
            ({**base, 'evidence': {}}, 'evidence_type'),
            ({**base, 'evidence': []}, 'evidence_empty'),
            ({**base, 'supported': False}, 'unsupported_content'),
            ({**base, 'evidence': [entry, entry]}, 'drug_duplicate'),
            ({**base, 'evidence': [entry, {}]}, 'evidence_fields'),
            ({**base, 'evidence': [entry, {**second, 'drug_evidence': '原文不存在的证据'}]},
             'drug_evidence_not_verbatim'),
        ]
        for answer, code in invalid_cases:
            with self.subTest(code=code):
                with self.assertRaises(GoldenValidationError) as caught:
                    validate(answer)
                self.assertEqual(caught.exception.code, code)
        # Even an uncited recommendation must have an indication available for audit.
        with self.assertRaises(GoldenValidationError) as caught:
            validate(base, introductions={'酮康唑': details['酮康唑']})
        self.assertEqual(caught.exception.code, 'indication_missing')
        # A serialized dialog substring spanning two utterances is not a valid quote.
        cross_turn = {**entry, 'case_evidence': '头皮屑伴瘙痒", "医生：请描述'}
        with self.assertRaises(GoldenValidationError) as caught:
            validate({**base, 'evidence': [cross_turn]},
                     dialog=json.dumps(['患者：头皮屑伴瘙痒', '医生：请描述'], ensure_ascii=False))
        self.assertEqual(caught.exception.code, 'case_evidence_not_verbatim')

    def test_golden_multi_source_citations_and_length_boundary(self):
        from agent.validation import GoldenValidationError, _validate_golden_experience
        base = json.loads(supported_golden())
        legacy = base['evidence'][0]
        indication = {'field': '适应症', 'quote': legacy['drug_evidence']}
        precaution = {'field': '注意事项', 'quote': '对本品过敏者禁用'}
        entry = {**legacy, 'case_evidence': [legacy['case_evidence'], '用药后仍然瘙痒'],
                 'drug_evidence': [indication, precaution]}
        args = dict(dialog=json.dumps([self.dialog, '患者：用药后仍然瘙痒']),
                    reasoning_summary='', drugs=['酮康唑'],
                    introductions={'酮康唑': {'适应症': indication['quote'],
                                             '注意事项': precaution['quote']}})
        answer = {**base, 'evidence': [entry], 'experience': '字' * 200}
        self.assertEqual(_validate_golden_experience(answer, **args), answer['experience'])
        invalid = [
            ({**answer, 'experience': '字' * 201}, 'experience_too_long'),
            ({**answer, 'evidence': [{**entry, 'case_evidence': []}]}, 'case_evidence_empty'),
            ({**answer, 'evidence': [{**entry, 'case_evidence': ['头皮屑伴瘙痒', '错误引用内容']}]},
             'case_evidence_not_verbatim'),
            ({**answer, 'evidence': [{**entry, 'case_evidence': ['[MASK]']}]}, 'case_evidence_mask'),
            ({**answer, 'evidence': [{**entry, 'drug_evidence': []}]}, 'drug_evidence_empty'),
            ({**answer, 'evidence': [{**entry, 'drug_evidence': [precaution]}]},
             'drug_evidence_indication_missing'),
            ({**answer, 'evidence': [{**entry, 'drug_evidence': [indication, {'field': '禁忌',
                                                                               'quote': precaution['quote']}]}]},
             'drug_evidence_source'),
            ({**answer, 'evidence': [{**entry, 'drug_evidence': [indication, {'field': '注意事项',
                                                                               'quote': indication['quote']}]}]},
             'drug_evidence_not_verbatim'),
            ({**answer, 'evidence': [{**entry, 'drug_evidence': [{'quote': indication['quote']}]}]},
             'drug_evidence_fields'),
        ]
        for value, code in invalid:
            with self.subTest(code=code):
                with self.assertRaises(GoldenValidationError) as caught:
                    _validate_golden_experience(value, **args)
                self.assertEqual(caught.exception.code, code)

    def test_positive_lesson_accepts_new_prompt_output(self):
        answer = json.loads(supported_golden())
        for entry in answer['evidence']:
            entry['case_evidence'] = [entry['case_evidence']]
            entry['drug_evidence'] = [{'field': '适应症', 'quote': entry['drug_evidence']}]
        fake = FakeLLMClient([recommendation('酮康唑'), json.dumps(answer, ensure_ascii=False)])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            self.addCleanup(session.close)
            case = load_dialmed_cases(self.dataset)[0]
            outcome = session.run_case(case)
            self.assertEqual(outcome['golden_experience'], answer['experience'])
        self.assertEqual(json.loads(self.golden.read_text()),
                         [{'case_id': case.case_id, 'experience': answer['experience']}])

    def test_positive_lesson_correction_and_attempts(self):
        invalid = json.loads(supported_golden())
        invalid['evidence'][0]['case_evidence'] = '原文不存在的证据'
        fake = FakeLLMClient([recommendation('酮康唑'), 'not json',
                              json.dumps(invalid, ensure_ascii=False), supported_golden()])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            self.addCleanup(session.close)
            outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
        self.assertIsNotNone(outcome['golden_experience'])
        self.assertEqual(len(fake.calls), 4)
        self.assertIn('校验未通过', fake.calls[2][1]['content'])
        self.assertIn('case_evidence 不是医患对话中的连续原文', fake.calls[3][1]['content'])
        self.assertEqual(fake.calls[3][1]['content'].count('上一版回复：'), 1)
        events = [json.loads(x) for x in (self.llm.parent/'diagnostics.jsonl').read_text().splitlines()]
        reviews = [x for x in events if x.get('stage') == 'golden_experience']
        self.assertEqual([x['status'] for x in reviews],
                         ['validation_rejected', 'validation_rejected', 'accepted'])
        self.assertEqual([x['attempt'] for x in reviews], [1, 2, 3])
        self.assertEqual(reviews[0]['raw_reply'], 'not json')
        self.assertIsNone(reviews[0]['parsed_reply'])
        self.assertEqual(reviews[0]['error_code'], 'reply_format_invalid')
        self.assertEqual(reviews[1]['parsed_reply'], invalid)
        self.assertEqual(reviews[1]['error_code'], 'case_evidence_not_verbatim')
        self.assertEqual(reviews[2]['raw_reply'], supported_golden())

    def test_positive_correction_stops_on_unsupported_or_api_error(self):
        for reply in (unsupported_golden(), TimeoutError('timeout'), ValueError('API format')):
            with self.subTest(reply=type(reply).__name__):
                fake = FakeLLMClient([recommendation('酮康唑'), 'not json', reply])
                with patch('main.LLMClient', return_value=fake):
                    session = self.session(training=True)
                    try:
                        outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
                    finally:
                        session.close()
                self.assertIsNone(outcome['golden_experience'])
                self.assertEqual(len(fake.calls), 3)

    def test_training_saves_lesson_without_using_it_in_next_doctor_prompt(self):
        second_dialog = '患者：头皮屑伴瘙痒四天'
        with self.dataset.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'dialog': [second_dialog], 'label': ['酮康唑']},
                                    ensure_ascii=False) + '\n')
        fake = FakeLLMClient([
            recommendation('酮康唑'), supported_golden(),
            recommendation('酮康唑'), unsupported_golden(),
        ])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
            session.run_case(DialMedCase(2, second_dialog, ('酮康唑',)))
            prompt = fake.calls[2][1]['content']
            self.assertNotIn('相关历史经验', prompt)
            self.assertNotIn(json.loads(supported_golden())['experience'], prompt)
            self.assertNotIn('golden_experiences', fake.calls[2][0]['content'])
            self.assertTrue(self.golden.exists())
            self.assertNotIn('case_id', prompt)
            self.assertNotIn('dialog_excerpt', prompt)
            self.assertNotIn('similarity', prompt)
            self.assertNotIn('actual_drugs', prompt)
            session.close()

    def test_doctor_receives_only_positive_and_negative_lesson_text(self):
        second_dialog = '患者：头皮屑伴脱屑四天'
        third_dialog = '患者：头皮屑伴瘙痒和脱屑五天'
        with self.dataset.open('a', encoding='utf-8') as stream:
            for dialog in (second_dialog, third_dialog):
                stream.write(json.dumps({'dialog': [dialog], 'label': ['酮康唑']},
                                        ensure_ascii=False) + '\n')
        positive = '头皮屑伴瘙痒时核对药物适应症。'
        negative = '脱屑时应核对病因，避免只按症状选药。'
        self.golden.write_text(json.dumps([{'case_id': 1, 'experience': positive}],
                                          ensure_ascii=False), encoding='utf-8')
        self.reflections.write_text(json.dumps([{'case_id': 2, 'reflection': negative}],
                                               ensure_ascii=False), encoding='utf-8')
        fake = FakeLLMClient([json.dumps(['酮康唑'], ensure_ascii=False)])
        with patch('main.LLMClient', return_value=fake):
            session = self.session(training=False)
            session.run_case(DialMedCase(3, third_dialog, ('酮康唑',)))
            prompt = fake.calls[0][1]['content']
            history = json.loads(prompt.split('相关历史经验：', 1)[1]
                                 .split('\n目标药物数量：', 1)[0])
            self.assertEqual(history, {
                'golden_experiences': [positive],
                'reflections': [negative],
            })
            for forbidden in ('case_id', 'summary', 'dialog_excerpt', 'similarity',
                              self.dialog, second_dialog, 'actual_drugs'):
                self.assertNotIn(forbidden, prompt)
            self.assertEqual(session.memories.last_retrieval_stats['returned_count'], 2)
            session.close()

    def test_success_timeout_does_not_block_case(self):
        fake = FakeLLMClient([recommendation('酮康唑'), TimeoutError('gateway timeout')])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
            self.assertTrue(outcome['exact_match'])
            self.assertIsNone(outcome['golden_experience'])
            self.assertFalse(self.golden.exists())
            self.assertFalse(self.history.exists())
            rows = [json.loads(row) for row in self.llm.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([row['record_type'] for row in rows],
                             ['llm_call', 'llm_call', 'case_total', 'golden_experience_total'])
            self.assertEqual(rows[1]['status'], 'timeout')
            session.close()

    def test_error_saves_reflection_and_does_not_leak_label_before_prediction(self):
        reflection = ('头皮屑伴瘙痒而病因未明时，应核对医生判断与药物适应症，'
                      '避免仅按共同症状替换药物。')
        fake = FakeLLMClient([
            recommendation('氯雷他定',
                           '关键证据：头皮屑伴瘙痒；疾病判断：考虑皮肤问题，仍需确认；'
                           '选药依据：按瘙痒症状选择对应药物。'),
            json.dumps({'reflection': reflection}, ensure_ascii=False),
        ])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
            self.assertFalse(outcome['exact_match'])
            self.assertEqual(outcome['reflection'], reflection)
            self.assertNotIn('酮康唑', fake.calls[0][1]['content'])
            self.assertIn('参考推荐药物：', fake.calls[1][1]['content'])
            row = json.loads(self.reflections.read_text(encoding='utf-8'))[0]
            self.assertEqual(row, {'case_id': 1, 'reflection': reflection})
            self.assertFalse(self.golden.exists())
            self.assertFalse(self.history.exists())
            self.assertFalse(self.reports.exists())
            session.close()

    def test_long_reflection_retry_reports_length_and_saves_valid_rewrite(self):
        long_text = '头皮屑伴瘙痒时应核对病程，避免仅凭症状推断病因。' * 12
        valid = '头皮屑伴瘙痒时，应核对病程及用药反馈，避免仅凭共同症状推断病因。'
        fake = FakeLLMClient([recommendation('酮康唑'),
                              json.dumps({'reflection': long_text}),
                              json.dumps({'reflection': valid})])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            try:
                outcome = session.run_case(DialMedCase(1, self.dialog, ('依巴斯汀',)))
                self.assertEqual(outcome['reflection'], valid)
                self.assertIn(f'{len(long_text)}个字符', fake.calls[2][1]['content'])
                self.assertIn('不设最低字数', fake.calls[2][1]['content'])
                self.assertIn('超过200字符上限', fake.calls[2][1]['content'])
                self.assertIn('证据不足时返回 {"reflection":""}', fake.calls[2][1]['content'])
                self.assertEqual(json.loads(self.reflections.read_text())[0]['reflection'], valid)
            finally:
                session.close()

    def test_unsupported_reflection_skips_retry_and_memory(self):
        for retries in (0, 1):
            with self.subTest(retries=retries):
                fake = FakeLLMClient([recommendation('酮康唑'),
                                      *['not json'] * retries, '{"reflection":""}'])
                with patch('main.LLMClient', return_value=fake):
                    session = self.session()
                    try:
                        outcome = session.run_case(DialMedCase(1, self.dialog, ('依巴斯汀',)))
                    finally:
                        session.close()
                self.assertIsNone(outcome['reflection'])
                self.assertNotIn('reflection_error', outcome)
                self.assertEqual(len(fake.calls), 2 + retries)
                self.assertFalse(self.reflections.exists())
                events = [json.loads(line) for line in
                          (self.llm.parent / 'diagnostics.jsonl').read_text().splitlines()]
                reviews = [event for event in events if event.get('stage') == 'reflection']
                self.assertEqual(reviews[-1]['status'], 'unsupported')
                saves = [event for event in events if event.get('stage') == 'memory_sync']
                self.assertEqual(saves[-1]['status'], 'no_new_lesson')

    def test_long_multisentence_reflection_is_saved_and_reloaded(self):
        reflection = (
            '头皮屑伴瘙痒且病因未明确时，应分别核对酮康唑与依巴斯汀的适应症及适用条件，'
            '不能仅凭二者涉及相同症状就认定可以替代，也不能仅因参考推荐不同认定医生选药错误。'
            '若病例材料未说明病因、既往用药及疗效反馈，需补充这些信息后判断取舍，'
            '不得将尚未报告的病史视为不存在，或为满足参考答案而编造其中一种药物具有疗效或安全性优势。'
        )
        self.assertGreater(len(reflection), 120)
        self.assertLessEqual(len(reflection), 200)
        fake = FakeLLMClient([recommendation('酮康唑'),
                              json.dumps({'reflection': reflection}, ensure_ascii=False)])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            try:
                outcome = session.run_case(DialMedCase(1, self.dialog, ('依巴斯汀',)))
                self.assertEqual(outcome['reflection'], reflection)
            finally:
                session.close()
        self.assertEqual(len(fake.calls), 2)
        reloaded = self.session()
        try:
            self.assertEqual(reloaded.memories._reflections[0]['reflection'], reflection)
        finally:
            reloaded.close()

    def test_reflection_length_type_and_multiple_drugs(self):
        from agent.validation import _validate_reflection
        args = dict(drugs=['药物甲', '药物乙', '药物丙'], dialog=self.dialog,
                    reasoning_summary='头皮屑伴瘙痒')
        text = ('头皮屑伴瘙痒时，应结合实际方案核对药物甲、药物乙与药物丙的联用限制，'
                '在证据不足时不能断言这些药物禁止合用。')
        _validate_reflection(text, **args)
        _validate_reflection('', **args)
        boundary = text + '待核实' * ((200 - len(text)) // 3)
        boundary += '。' * (200 - len(boundary))
        _validate_reflection(boundary, **args)
        for invalid in (boundary + '字', None, [], '   ', text + '\n另一个段落'):
            with self.subTest(invalid=repr(invalid)[:50]):
                with self.assertRaises(ValueError):
                    _validate_reflection(invalid, **args)

    def test_reflection_validation_does_not_audit_semantics(self):
        from agent.validation import _validate_reflection
        for lesson in ('医生认为需要进一步评估。', '标准标签不代表唯一正确。',
                       '由专业人员结合完整材料作进一步判断。', '药物甲'):
            with self.subTest(lesson=lesson):
                _validate_reflection(lesson, ['药物甲'], dialog='患者：头皮屑反复',
                                     reasoning_summary='头皮屑伴瘙痒')

    def test_repeated_invalid_reflection_keeps_outcome_and_session_usable(self):
        invalid = json.dumps({'reflection': '头皮屑伴瘙痒时应核对病程。' * 25})
        fake = FakeLLMClient([recommendation('酮康唑'), invalid, invalid, invalid,
                              recommendation('酮康唑'), unsupported_golden()])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            try:
                outcome = session.run_case(DialMedCase(1, self.dialog, ('依巴斯汀',)))
                self.assertFalse(outcome['exact_match'])
                self.assertIsNone(outcome['reflection'])
                self.assertIn('连续 3 次', outcome['reflection_error'])
                self.assertFalse(self.reflections.exists())
                self.assertTrue(all(agent.state == State.IDLE for agent in session.agents))
                next_outcome = session.run_case(DialMedCase(2, self.dialog, ('酮康唑',)))
                self.assertTrue(next_outcome['exact_match'])
                self.assertNotIn('reflection_error', next_outcome)
            finally:
                session.close()

    def test_test_stage_freezes_experience_and_writes_no_memory(self):
        self.golden.write_text(json.dumps([{
            'case_id': 1, 'experience': '旧经验'
        }], ensure_ascii=False), encoding='utf-8')
        fake = FakeLLMClient([json.dumps(['酮康唑'], ensure_ascii=False), supported_golden()])
        with patch('main.LLMClient', return_value=fake):
            session = self.session(training=False)
            self.golden.write_text(json.dumps([{
                'case_id': 1, 'experience': '外部新经验'
            }], ensure_ascii=False), encoding='utf-8')
            outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
            self.assertTrue(outcome['exact_match'])
            self.assertEqual(session.memories._golden[1], '旧经验')
            self.assertEqual(json.loads(self.golden.read_text(encoding='utf-8'))[0]['experience'],
                             '外部新经验')
            self.assertFalse(self.reflections.exists())
            self.assertFalse(self.reports.exists())
            session.close()

    def test_test_stage_skips_expert_generation_for_matches_and_mismatches(self):
        fake = FakeLLMClient([json.dumps(['酮康唑'], ensure_ascii=False), json.dumps(['酮康唑'], ensure_ascii=False)])
        with patch('main.LLMClient', return_value=fake):
            session = self.session(training=False)
            try:
                with patch.object(session.expert, 'ask_llm', side_effect=AssertionError('测试不应调用专家模型')), \
                     patch.object(session.medicines, 'introductions_for', side_effect=AssertionError('测试不应查药物说明')):
                    for case_id, actual in enumerate(('酮康唑', '依巴斯汀'), 1):
                        outcome = session.run_case(DialMedCase(case_id, self.dialog, (actual,)))
                        self.assertEqual(outcome['exact_match'], actual == '酮康唑')
                        self.assertIsNone(outcome['golden_experience'])
                        self.assertIsNone(outcome['reflection'])
                        self.assertTrue(all(agent.state == State.IDLE for agent in session.agents))
                capabilities = {packet.payload.get('capability') for packet in session.trace.packets}
                self.assertIn('retrieval.history', capabilities)
                self.assertNotIn('retrieval.medicines', capabilities)
                self.assertNotIn('expert.review', capabilities)
                self.assertEqual(len(fake.calls), 2)
                rows = [json.loads(line) for line in self.llm.read_text().splitlines()]
                self.assertEqual([row['record_type'] for row in rows],
                                 ['llm_call', 'case_total', 'llm_call', 'case_total'])
                self.assertTrue(all(row['agent'] == 'DoctorAgent' for row in rows
                                    if row['record_type'] == 'llm_call'))
                self.assertFalse(self.golden.exists())
                self.assertFalse(self.reflections.exists())
            finally:
                session.close()

    def test_doctor_retries_invalid_output_and_keeps_known_count(self):
        fake = FakeLLMClient([
            json.dumps(['无效药'], ensure_ascii=False), json.dumps(['酮康唑'], ensure_ascii=False),
        ])
        with patch('main.LLMClient', return_value=fake):
            session = self.session(training=False)
            outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
            self.assertTrue(outcome['exact_match'])
            self.assertEqual(len(fake.calls), 2)
            self.assertIn('恰好输出 1 个', fake.calls[0][0]['content'])
            session.close()

    def test_expert_timeout_is_recorded_without_stale_token_usage(self):
        fake = FakeLLMClient([recommendation('酮康唑'), TimeoutError('test timeout')])
        with patch('main.LLMClient', return_value=fake):
            session = self.session()
            try:
                outcome = session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
                self.assertIsNone(outcome['golden_experience'])
            finally:
                session.close()
        events = [json.loads(line) for line in (self.llm.parent/'diagnostics.jsonl').read_text().splitlines()]
        self.assertTrue(any(e['event'] == 'model_call_started' for e in events))
        call = next(e for e in events if e['event'] == 'model_call_finished'
                    and e['agent'] == 'ExpertAgent')
        self.assertEqual(call['status'], 'timeout')
        self.assertFalse(call['usage_known'])
        self.assertEqual(call['total_tokens'], 0)
        self.assertGreaterEqual(call['elapsed_ms'], 0)
        self.assertTrue(any(e.get('stage') == 'golden_experience' and e['status'] == 'timeout' for e in events))
        totals = [json.loads(line) for line in self.llm.read_text().splitlines()
                  if json.loads(line)['record_type'] == 'case_total']
        self.assertFalse(totals[-1]['usage_complete'])

    def test_expert_unsupported_and_invalid_are_distinct(self):
        for reply, expected in ((unsupported_golden(), 'unsupported'), ('{"supported": "yes"}', 'validation_rejected')):
            with self.subTest(expected=expected):
                attempts = 3 if expected == 'validation_rejected' else 1
                fake = FakeLLMClient([recommendation('酮康唑'), *[reply] * attempts])
                with patch('main.LLMClient', return_value=fake):
                    session = self.session()
                    try:
                        session.run_case(DialMedCase(1, self.dialog, ('酮康唑',)))
                    finally:
                        session.close()
                events = [json.loads(line) for line in (self.llm.parent/'diagnostics.jsonl').read_text().splitlines()]
                reviews = [e for e in events if e.get('stage') == 'golden_experience']
                self.assertEqual(reviews[-1]['status'], expected)


if __name__ == '__main__':
    unittest.main()
