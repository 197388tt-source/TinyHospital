"""Qwen Doctor prompt and response contracts, without model calls."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.doctor_Qwen import DoctorAgent
from tool.tracing import Trace


class QwenDoctorPromptTests(unittest.TestCase):
    reply = json.dumps({
        'recommended_drugs': ['药物甲'],
        'reasoning_summary': '关键证据：头痛三天；疾病判断：病因待明确；选药依据：用于缓解当前头痛症状。',
    }, ensure_ascii=False)

    def doctor(self, training=False):
        return DoctorAgent(Trace(verbose=False), SimpleNamespace(names=('药物甲', '药物乙')),
                           lambda: object(), training=training)

    def request(self, history=None):
        data = {'case_id': 1, 'count': 1, 'patient_dialog': '患者：头痛三天'}
        if history is not None:
            data['history'] = history
        return json.dumps(data, ensure_ascii=False)

    def test_baseline_matches_training_for_all_counts(self):
        doctor = self.doctor()
        for count in (1, 2):
            self.assertEqual(doctor._baseline_instruction(count), doctor._training_instruction(count))

    def test_test_stage_parses_objects_with_or_without_history(self):
        for history in (None, {'golden_experiences': ['经验'], 'reflections': ['反思']}):
            with self.subTest(history=history):
                doctor = self.doctor()
                with patch.object(doctor, 'ask_llm', return_value=self.reply) as call:
                    result = json.loads(doctor.recommend(self.request(history)))
                self.assertEqual(result, {'case_id': 1, 'recommended_drugs': ['药物甲']})
                instruction = call.call_args.args[1]
                self.assertTrue(instruction.startswith(doctor._training_instruction(1)))
                if history:
                    self.assertIn('2条药物推荐经验', instruction)
                    self.assertIn('相关历史经验', call.call_args.args[2])

    def test_test_stage_retries_arrays_and_invalid_summary(self):
        invalid_summary = json.dumps({'recommended_drugs': ['药物甲'], 'reasoning_summary': ''})
        doctor = self.doctor()
        with patch.object(doctor, 'ask_llm', side_effect=['["药物甲"]', invalid_summary, self.reply]) as call:
            result = json.loads(doctor.recommend(self.request()))
        self.assertEqual(call.call_count, 3)
        self.assertEqual(result['recommended_drugs'], ['药物甲'])
        self.assertIn('上一回复无效', call.call_args.args[2])

    def test_training_still_returns_summary(self):
        doctor = self.doctor(training=True)
        with patch.object(doctor, 'ask_llm', return_value=self.reply):
            result = json.loads(doctor.recommend(self.request()))
        self.assertEqual(result['reasoning_summary'], json.loads(self.reply)['reasoning_summary'])


if __name__ == '__main__':
    unittest.main()
