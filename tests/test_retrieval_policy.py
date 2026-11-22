"""Threshold/diagnostic regression checks without model calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from history.memory import CaseMemoryStore


class RetrievalPolicyTests(unittest.TestCase):
    def test_embedding_cutoff_and_candidate_reasons(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, golden, reflection = root/'train.txt', root/'golden.json', root/'reflection.json'
            dataset.write_text('\n'.join(json.dumps({'dialog': [x]}) for x in ('鼻窦炎', '过敏皮疹')))
            golden.write_text('[]')
            reflection.write_text(json.dumps([{'case_id': 1, 'reflection': '鼻部经验'},
                                             {'case_id': 2, 'reflection': '皮疹经验'}]))
            with patch('embedding.retrieval.SemanticMemoryIndex') as factory:
                index = factory.return_value
                index.identity.side_effect = lambda bucket, row: str(row['case_id'])
                index.scores.return_value = {'1': 0.37, '2': 0.55}
                store = CaseMemoryStore(golden, reflection, False, dataset,
                                        embedding_options={'diagnostics_path': root/'diagnostics.jsonl'})
                self.assertEqual(store.retrieve('手癣', 3, str(dataset))['reflections'], [])
                self.assertEqual(store.last_retrieval_stats['min_similarity'], 0.60)
                self.assertEqual([c['reason'] for c in store.last_retrieval_stats['candidates']],
                                 ['below_threshold', 'below_threshold'])
                index.scores.return_value = {'1': 0.37, '2': 0.80}
                self.assertEqual(store.retrieve('皮肤过敏', 3, str(dataset))['reflections'][0]['case_id'], 2)
                self.assertEqual(store.retrieve('皮肤过敏', 2, str(dataset))['reflections'], [])
                self.assertEqual(store.last_retrieval_stats['candidates'][1]['reason'], 'current_case')
                events = [json.loads(line) for line in (root/'diagnostics.jsonl').read_text().splitlines()]
                self.assertEqual(len(events), 3)
                self.assertEqual(events[-1]['event'], 'retrieval')
                self.assertNotIn('dialog', events[-1])
            jaccard = CaseMemoryStore(golden, reflection, False, dataset, retrieval_mode='jaccard')
            self.assertEqual(jaccard.min_similarity, 0.15)

    def test_invalid_threshold_fails_before_model_use(self):
        for threshold in (-1, 1.1, True, float('nan')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                CaseMemoryStore(Path('missing1'), Path('missing2'), False,
                                embedding_options={'min_similarity': threshold})
