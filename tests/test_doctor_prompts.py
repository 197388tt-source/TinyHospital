"""Stage-specific model inputs and output contracts, without live API calls."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.doctor import DoctorAgent
from tests.test_main import FakeLLMClient, recommendation
from tool.medicine_store import MedicineKnowledgeBase
from tool.tracing import Trace


class DoctorPromptTests(unittest.TestCase):
    def doctor(self, replies, training):
        model = FakeLLMClient(replies)
        doctor = DoctorAgent(Trace(verbose=False), MedicineKnowledgeBase(),
                             lambda: model, training=training)
        doctor.trace.message_costs.append(SimpleNamespace(step=4))
        return doctor, model

    def request(self, **extra):
        return json.dumps({'case_id': 1, 'count': 1,
                           'patient_dialog': '["患者：头皮屑伴瘙痒三天"]', **extra},
                          ensure_ascii=False)

    def test_training_ignores_supplied_history_including_retries(self):
        doctor, model = self.doctor(['[]', recommendation('酮康唑')], training=True)
        history = {'golden_experiences': ['正向经验独有内容'], 'reflections': ['反思独有内容']}
        result = json.loads(doctor.recommend(self.request(history=history)))
        self.assertEqual(result['recommended_drugs'], ['酮康唑'])
        self.assertTrue(result['reasoning_summary'])
        self.assertEqual(len(model.calls), 2)
        for messages in model.calls:
            text = ''.join(message['content'] for message in messages)
            self.assertIn('头皮屑伴瘙痒三天', text)
            self.assertIn('reasoning_summary', text)
            for forbidden in ('正向经验独有内容', '反思独有内容', 'golden_experiences', 'reflections', '历史经验'):
                self.assertNotIn(forbidden, text)

    def test_training_needs_no_history_but_requires_reasoning(self):
        doctor, model = self.doctor([
            '{"recommended_drugs":["酮康唑"]}', recommendation('酮康唑'),
        ], training=True)
        result = json.loads(doctor.recommend(self.request()))
        self.assertTrue(result['reasoning_summary'])
        self.assertEqual(len(model.calls), 2)

    def test_test_stage_accepts_arrays_with_both_one_or_no_lesson_types(self):
        for positive, negative in (([], []), (['正向经验'], []), ([], ['错误反思']),
                                   (['正向经验'], ['错误反思'])):
            with self.subTest(positive=positive, negative=negative):
                doctor, model = self.doctor(['["酮康唑"]'], training=False)
                history = {'golden_experiences': positive, 'reflections': negative}
                with patch('agent.doctor._validate_reasoning', side_effect=AssertionError('测试不校验依据')):
                    result = json.loads(doctor.recommend(self.request(history=history)))
                self.assertEqual(result, {'case_id': 1, 'recommended_drugs': ['酮康唑']})
                system, user = (message['content'] for message in model.calls[0])
                self.assertTrue(system.startswith('你是一名医生，你的任务是根据下面的医患对话进行药物推荐。'))
                self.assertIn('只输出 JSON 字符串数组', system)
                self.assertNotIn('reasoning_summary', system)
                if positive or negative:
                    self.assertIn('两类也可能同时为空', system)
                    self.assertIn('均不是当前医患对话原文', system)
                    self.assertEqual(json.loads(user.split('\n相关历史经验：', 1)[1]), history)
                else:
                    self.assertEqual(user, json.loads(self.request())['patient_dialog'])
                    self.assertEqual(system, (
                        '你是一名医生，你的任务是根据下面的医患对话进行药物推荐。'
                        f"恰好输出 1 个药名，必须逐字从以下白名单中选择：{'、'.join(doctor.medicines.names)}。"
                        '只输出 JSON 字符串数组，例如 ["药名一","药名二"]；不要解释、编号或 Markdown。'
                    ))

    def test_test_stage_rejects_old_object_and_invalid_drug_lists(self):
        for invalid in (recommendation('酮康唑'), '[]', '["白名单外药物"]', '[42]',
                        '["酮康唑","酮康唑"]'):
            with self.subTest(invalid=invalid):
                doctor, model = self.doctor([invalid, '["酮康唑"]'], training=False)
                result = json.loads(doctor.recommend(self.request()))
                self.assertEqual(result['recommended_drugs'], ['酮康唑'])
                self.assertNotIn('reasoning_summary', result)
                self.assertEqual(len(model.calls), 2)
                self.assertEqual(model.calls[0][0], model.calls[1][0])
                for messages in model.calls:
                    text = ''.join(message['content'] for message in messages)
                    for forbidden in ('历史经验', 'golden_experiences', 'reflections'):
                        self.assertNotIn(forbidden, text)
