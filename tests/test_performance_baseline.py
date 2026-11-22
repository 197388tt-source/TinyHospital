import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases import DIALMED_TRAIN_PATH, DialMedCase, load_dialmed_cases  # noqa: E402
from main import HospitalSession  # noqa: E402
from performance_baseline import FixedModel, ROUTE, measure  # noqa: E402
from performance_stage2 import BASELINE, compare  # noqa: E402


class MedicineDeduplicationTests(unittest.TestCase):
    def test_step7_transmits_each_description_once_for_full_and_partial_overlap(self):
        dialog = json.dumps(['患者：头皮屑伴瘙痒三天'], ensure_ascii=False)
        cases = [DialMedCase(1, dialog, ('酮康唑',)),
                 DialMedCase(2, dialog, ('酮康唑', '硫磺皂'))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('main.LLMClient', return_value=FixedModel()):
                session = HospitalSession(
                    retrieval_mode='jaccard', verbose=False, training=True, dataset_path=DIALMED_TRAIN_PATH,
                    golden_path=root / 'golden.json', reflection_path=root / 'reflections.json',
                    communication_path=root / 'traffic.jsonl', control_path=root / 'control.jsonl',
                    llm_path=root / 'llm.jsonl',
                )
                try:
                    for case in cases:
                        session.run_case(case)
                finally:
                    session.close()
            responses = [packet for packet in session.trace.packets
                         if packet.payload.get('capability') == 'expert.review']
            self.assertEqual(len(responses), 2)
            for packet in responses:
                body = json.loads(packet.payload['text'])
                self.assertEqual(set(body), {'case_id', 'recommended_drugs', 'actual_drugs',
                                             'medicine_introductions'})
                names = list(dict.fromkeys(body['recommended_drugs'] + body['actual_drugs']))
                self.assertEqual(list(body['medicine_introductions']), names)
                self.assertEqual(body['medicine_introductions'],
                                 session.medicines.introductions_for(names))
            self.assertEqual(len(set(json.loads(responses[0].payload['text'])['recommended_drugs']) &
                                 set(json.loads(responses[0].payload['text'])['actual_drugs'])), 1)
            self.assertEqual(len(set(json.loads(responses[1].payload['text'])['recommended_drugs']) &
                                 set(json.loads(responses[1].payload['text'])['actual_drugs'])), 1)

    def test_ten_fixed_cases_complete_with_timing_and_label_isolation(self):
        result = measure()
        self.assertEqual(result['case_ids'], list(range(1, 11)))
        self.assertEqual(result['packet_count'], 80)
        self.assertEqual(len(result['steps']), len(ROUTE))
        self.assertTrue(result['label_isolation_checked'])
        self.assertIsNone(result['real_llm_api_tokens'])
        self.assertEqual(result['dialog_transmitted_bytes_steps_1_3_4'],
                         result['dialog_unique_bytes'] * 2)
        self.assertEqual(result['history_nonempty_cases'], 0)
        self.assertEqual(result['duplicate_medicine_introduction_bytes_step_7'], 0)
        names = FixedModel().names
        expected_overlap = sum(len(set(names[:case.drug_count]) & set(case.drug_labels))
                               for case in load_dialmed_cases(DIALMED_TRAIN_PATH, 10))
        self.assertEqual(sum(item['overlap_count'] for item in result['step7_case_bytes']), expected_overlap)
        self.assertTrue(all(item['current_packet_bytes'] < item['legacy_packet_bytes_reconstructed']
                            for item in result['step7_case_bytes'] if item['overlap_count']))
        for step in result['steps']:
            self.assertGreater(step['mean_execution_ms'], 0)
            self.assertGreater(step['mean_serialization_ms'], 0)
            self.assertLessEqual(step['mean_serialization_ms'], step['mean_execution_ms'])
            self.assertGreater(step['mean_packet_bytes'], step['mean_message_bytes'])

    @unittest.skipUnless(BASELINE.exists(), '先前按要求清空 runs，第一阶段冻结基线文件不存在')
    def test_saved_first_stage_baseline_matches(self):
        comparison = compare()
        self.assertTrue(comparison['introduction_hashes_match'])
        self.assertTrue(comparison['prompt_hashes_match'])


if __name__ == '__main__':
    unittest.main()
