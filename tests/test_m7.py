import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bluecad_cost.m7 import (CONDITIONS, TASKS, cell, correct, fixes, groups, overlap, seeds, taxonomy, verify)
from bluecad_cost.scoring import parse_prediction, score

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

class M7Tests(unittest.TestCase):
    def test_taxonomy_parser_precedence(self):
        for raw, expected in [('', 'no_letter'),('Yes or No', 'no_letter'),('E','letter_out_of_range'),('A and E','multiple_letters'),('ANSWER: B','parsed_correct'),('choice C','parsed_but_wrong'),('A and B','parsed_correct')]:
            self.assertEqual(taxonomy(raw,'B'), expected)
            self.assertEqual(expected.startswith('parsed_'), parse_prediction(raw,'2') is not None)

    def test_overlap_groups_and_fix_break(self):
        items=[{'id':str(i),'answer':'yes','group':'x' if i<2 else 'y'} for i in range(3)]
        a={str(i):{'pred':p} for i,p in enumerate(['no','no','yes'])}
        b={str(i):{'pred':p} for i,p in enumerate(['no','yes','no'])}
        self.assertEqual(overlap(items,{'a':a,'b':b},'task1'),{'all_failed':1,'only_one_failed':2,'none_failed':0})
        self.assertEqual(fixes(items,a,b,'task1')['fixed'],['1'])
        self.assertEqual(fixes(items,a,b,'task1')['broken'],['2'])
        self.assertEqual(groups(items)['group'],{'0':'x','1':'x','2':'y'})
        subset=[x for x in items if groups(items)['group'][x['id']]=='x']
        self.assertEqual(sum(correct(x,b[x['id']],'task1') for x in subset)/len(subset), .5)

    def test_verification_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'results'; official=Path(tmp)/'items'; official.mkdir()
            expected={c:((1,)*len(seeds(c)),)*3 for c in CONDITIONS}
            for task in TASKS:
                answer='A' if task=='task2' else 'yes'
                items=[{'id':str(i),'answer':answer} for i in range(2)]
                (official/f'{task}_items.json').write_text(json.dumps({'items':items}))
                for cond in CONDITIONS:
                    for seed in seeds(cond):
                        directory=cell(root,cond,seed); directory.mkdir(parents=True,exist_ok=True)
                        pred=[{'id':x['id'],'pred':answer} for x in items]
                        (directory/f'{task}_predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pred))
                        (directory/f'{task}_metrics.json').write_text(json.dumps(score(items,pred,'2' if task=='task2' else '1')))
            verify(root,official,count=2,expected=expected)
            path=cell(root,'siglip2',0)/'task1_predictions.jsonl'
            path.write_text('{"id":"0","pred":"yes"}\n'*2)
            with self.assertRaisesRegex(ValueError,'coverage'): verify(root,official,count=2,expected=expected)
            path.write_text('{"id":"0","pred":"yes"}\n{"id":"1","pred":"no"}\n')
            with self.assertRaisesRegex(ValueError,'mismatch'): verify(root,official,count=2,expected=expected)

    def test_cost_missing_and_cheapest(self):
        mod=module('cost_reading_m7')
        rows=[dict(task='task1',condition='a',accuracy_all_mean='.9',gpu_hours_mean=''),dict(task='task1',condition='b',accuracy_all_mean='.895',gpu_hours_mean='2'),dict(task='task1',condition='c',accuracy_all_mean='.8',gpu_hours_mean='1')]
        result=mod.reading(rows,'task1')
        self.assertEqual((result['condition'],result['eligible_missing_cost'],result['status']),('b','a','incomplete_costs'))

    def test_plot_empty_cumulative_is_optional(self):
        # Importing plot script must not require matplotlib.
        mod=module('plot_m7')
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            (base/'main_table.csv').write_text('task,condition,gpu_hours_mean,accuracy_all_mean,accuracy_all_std,protocol\ntask1,siglip2,1,0.5,0.1,3 seeds\n')
            (base/'learning_curves.csv').write_text('condition,train_size,task1_accuracy_all,task1_dim_accuracy_all,task2_accuracy_all\nsiglip2,2000,0.5,0.4,0.3\n')
            axes=[MagicMock() for _ in range(3)]
            pyplot=MagicMock(); pyplot.subplots.side_effect=[(MagicMock(),axes),(MagicMock(),axes)]
            matplotlib=MagicMock(); matplotlib.pyplot=pyplot
            with patch.dict(sys.modules, {'matplotlib':matplotlib, 'matplotlib.pyplot':pyplot}):
                mod.plot(base,base/'figures')
            self.assertEqual(pyplot.subplots.call_count,2)

if __name__=='__main__': unittest.main()
