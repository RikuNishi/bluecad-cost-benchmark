import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bluecad_cost.scoring import parse_prediction


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


eval_script = module('run_qwen_eval')
builder = module('build_qwen_train_jsonl')
train = module('train_qwen_lora')


class M4Tests(unittest.TestCase):
    def test_prompts(self):
        self.assertIn('rendered view', eval_script.resolve_prompt({'prompt_key': 'task1_verification'}))
        item = {'prompt_key': 'task3_retrieval', 'prompt_vars': {'n': 4, 'labels': 'A, B, C, D'}}
        prompt = eval_script.resolve_prompt(item)
        self.assertIn('following 4 images', prompt)
        self.assertIn('labeled A, B, C, D', prompt)
        item['prompt_key'] = 'task2_retrieval'
        self.assertEqual(prompt, eval_script.resolve_prompt(item))

    def test_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, output = Path(tmp) / 'in.jsonl', Path(tmp) / 'out.jsonl'
            rows = [{'drawing_path': 'missing/d', 'render_path': 'missing/r', 'label': n} for n in (1, 0, 1, 0)]
            manifest.write_text(''.join(json.dumps(x) + '\n' for x in rows))
            self.assertEqual(builder.build(manifest, output, check_paths=False), 4)
            actual = [json.loads(x) for x in output.read_text().splitlines()]
            self.assertEqual([x['answer'] for x in actual], ['Yes', 'No', 'Yes', 'No'])
            self.assertEqual(actual[0]['drawing_path'], 'missing/d')
            with self.assertRaises(FileNotFoundError):
                builder.build(manifest, output)

    def test_lora_and_messages(self):
        self.assertEqual((train.LORA['r'], train.LORA['lora_alpha'], train.LORA['lora_dropout']), (16, 32, .05))
        self.assertEqual(train.LORA['target_modules'], ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'qkv', 'proj', 'linear_fc1', 'linear_fc2'])
        self.assertEqual(train.LORA['bias'], 'none')
        self.assertEqual(train.LORA['task_type'], 'CAUSAL_LM')
        msgs = train.messages({'drawing_path': 'd', 'render_path': 'r', 'answer': 'Yes'}, True)
        self.assertEqual([x['image'] for x in msgs[1]['content'][:2]], ['d', 'r'])
        self.assertEqual(msgs[2]['content'][0]['text'], 'Yes')

    def test_parsing(self):
        self.assertEqual(parse_prediction('Yes, same part.', '1'), 'yes')
        self.assertEqual(parse_prediction('Answer: (C)', '2'), 'C')
        self.assertIsNone(parse_prediction('uncertain', '1'))

if __name__ == '__main__':
    unittest.main()
