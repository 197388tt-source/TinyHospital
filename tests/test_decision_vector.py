import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_vector import DecisionVectorCodec  # noqa: E402
from ipc_runtime import _pack_binary_message, _unpack_binary_message  # noqa: E402


class DecisionVectorTests(unittest.TestCase):
    def setUp(self):
        self.codec = DecisionVectorCodec()

    def test_roundtrip_preserves_catalog_ids_and_original_order(self):
        names = list(self.codec.names)
        for selected in ([names[0]], [names[-1], names[0], names[17]], list(reversed(names))):
            with self.subTest(selected=selected[:3]):
                raw = self.codec.encode(selected)
                self.assertIs(type(raw), bytes)
                self.assertEqual(self.codec.decode(raw), selected)
                self.assertEqual(len(raw), 16 + self.codec.vector_bytes + len(selected))
                for name in selected:
                    number = self.codec.indices[name]
                    self.assertTrue(raw[16 + number // 8] & (1 << (number % 8)))

    def test_rejects_unknown_duplicate_and_corrupt_state(self):
        names = self.codec.names
        for bad in ([], [names[0], names[0]], ['未知药物'], '药物'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.codec.encode(bad)
        good = self.codec.encode([names[0], names[1]])
        damaged = [b'', good[:-1], b'FAIL' + good[4:],
                   good[:4] + b'\x02' + good[5:],
                   good[:8] + b'X' + good[9:],
                   good[:-1] + good[-2:-1],
                   good[:16] + bytes([good[16] & ~2]) + good[17:]]
        for raw in damaged:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.codec.decode(raw)
        with self.assertRaises(ValueError):
            self.codec.decode('不是 bytes')

    def test_catalog_fingerprint_prevents_mismatched_labels(self):
        names = list(self.codec.names)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'medicine.json'
            swapped = names[:]
            swapped[0], swapped[1] = swapped[1], swapped[0]
            path.write_text(json.dumps({name: i for i, name in enumerate(swapped)},
                                       ensure_ascii=False), encoding='utf-8')
            other = DecisionVectorCodec(path)
            with self.assertRaisesRegex(ValueError, '目录'):
                other.decode(self.codec.encode([names[0]]))

    def test_pipe_envelope_contains_raw_binary_not_json_or_base64(self):
        binary = self.codec.encode([self.codec.names[0]])
        wire = b'{"head":"test"}'
        envelope = _pack_binary_message(wire, binary)
        self.assertTrue(envelope.startswith(b'\x00THDV'))
        self.assertIn(binary, envelope)
        self.assertEqual(_unpack_binary_message(envelope), (wire, binary))
        self.assertEqual(_unpack_binary_message(wire), (wire, None))
        with self.assertRaises(ValueError):
            _unpack_binary_message(b'\x00THDV\x00')


if __name__ == '__main__':
    unittest.main()
