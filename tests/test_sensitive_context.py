import copy
import unittest

from sensitive_context.build import build, annotated
from sensitive_context.schema import messages, validate
from sensitive_context.train_lora import encode
from sensitive_context.replay import convert
from sensitive_context.evaluate import summarize
from sensitive_context.compare import aligned
from sensitive_context.report import public_report
from sensitive_context.evaluate_gliner import source_spans


class CharacterTokenizer:
    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        result=''.join(m['role']+':'+m['content']+'\n' for m in chat)
        return result+('assistant:' if add_generation_prompt else '')
    def __call__(self,text,add_special_tokens=False):return {'input_ids':[ord(c)+1 for c in text]}


class SensitiveContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=build(3)

    def test_splits_and_substitutions_do_not_overlap(self):
        families={};hashes={}
        for split,rows in self.rows.items():
            for row in rows:
                self.assertEqual(families.setdefault(row['family'],split),split)
                self.assertNotIn(row['sha256'],hashes)
                hashes[row['sha256']]=split
                if split!='review':validate(row)

    def test_sensitive_range_can_be_identical_to_benign_range(self):
        rows=[r for r in self.rows['test'] if r['family']=='transcript_two_contexts']
        positive=next(r for r in rows if r['spans'])
        s=positive['spans'][0];value=positive['text'][s['start']:s['end']]
        self.assertEqual(positive['text'].count(value),2)
        self.assertGreater(s['start'],positive['text'].index(value))

    def test_repeated_secret_has_two_distinct_offsets(self):
        row=next(r for r in self.rows['test'] if r['family']=='two_logins' and r['spans'])
        self.assertEqual(len(row['spans']),2)
        self.assertLess(row['spans'][0]['end'],row['spans'][1]['start'])

    def test_only_completion_tokens_receive_loss(self):
        row=self.rows['train'][0];tok=CharacterTokenizer()
        encoded=encode(row,tok,10000)
        prefix=tok.apply_chat_template(row['messages'][:-1],add_generation_prompt=True)
        n=len(prefix)
        self.assertEqual(encoded['labels'][:n],[-100]*n)
        self.assertEqual(encoded['labels'][n:],encoded['input_ids'][n:])
        self.assertTrue(encoded['labels'][n:])

    def test_truncation_is_rejected(self):
        with self.assertRaises(OverflowError):encode(self.rows['train'][0],CharacterTokenizer(),10)

    def test_test_and_ambiguous_rows_cannot_enter_training(self):
        with self.assertRaises(ValueError):encode(self.rows['test'][0],CharacterTokenizer(),10000)
        with self.assertRaises(ValueError):encode(self.rows['review'][0],CharacterTokenizer(),10000)

    def test_chat_and_annotation_cannot_disagree(self):
        row=copy.deepcopy(self.rows['train'][0]);row['messages'][2]['content']=row['text']
        with self.assertRaises(ValueError):validate(row)

    def test_legacy_malformed_target_is_rejected(self):
        row={'source':'gravitee','messages':[{'role':'system','content':'tag'},
             {'role':'user','content':'Entity types:\n- password\n\nText:\nabcd'},
             {'role':'assistant','content':'<password>abcd'}]}
        with self.assertRaises(ValueError):convert(row,0)

    def test_literal_markup_collision_is_rejected(self):
        row=copy.deepcopy(self.rows['train'][0]);row.pop('messages');row['text']='<password>literal</password>';row['spans']=[]
        with self.assertRaises(ValueError):validate(row)

    def test_empty_and_whitespace_spans_are_rejected(self):
        for text,start,end in [('abcd',1,1),('a  b',1,3)]:
            row={'text':text,'entities':['password'],'spans':[dict(start=start,end=end,label='password')]}
            with self.assertRaises(ValueError):validate(row)

    def test_errors_are_not_benign_passes(self):
        rows=[{'expected_decision':'sensitive','expected_spans':[(0,2,'password')],'detected':None,'error':'timeout'},
              {'expected_decision':'benign','expected_spans':[],'detected':None,'error':'timeout'},
              {'expected_decision':'review_required','expected_spans':[],'detected':False}]
        s=summarize(rows)
        self.assertEqual(s['scored_cases'],2);self.assertEqual(s['review_cases'],1)
        self.assertEqual(s['sensitive_document_recall'],0);self.assertEqual(s['span_fn'],1)
        self.assertEqual(s['benign_false_block_rate'],0);self.assertEqual(s['benign_total_withhold_rate'],1)
        self.assertEqual(s['inspection_errors'],2)

    def test_comparison_requires_same_gold_and_runtime(self):
        original={'status':'complete','runtime':{'data_sha256':'x','device':'cuda','effective_rope':{'rope_theta':1000000,'rope_scaling':None}},'results':[
            {'id':'one','sha256':'s','expected_decision':'benign','expected_spans':[],'category':'api_key','family':'f'}]}
        aligned(original,copy.deepcopy(original))
        changed=copy.deepcopy(original);changed['runtime'].pop('effective_rope')
        with self.assertRaises(ValueError):aligned(changed,copy.deepcopy(changed))
        changed=copy.deepcopy(original);changed['runtime']['effective_rope']['rope_theta']=10000
        with self.assertRaises(ValueError):aligned(changed,copy.deepcopy(changed))
        changed=copy.deepcopy(original);changed['results'][0]['expected_decision']='sensitive'
        with self.assertRaises(ValueError):aligned(original,changed)
        changed=copy.deepcopy(original);changed['runtime']['device']='mps'
        with self.assertRaises(ValueError):aligned(original,changed)
        changed=copy.deepcopy(original);changed['runtime']['decoder_sha256']='changed-decoder'
        with self.assertRaises(ValueError):aligned(original,changed)
        changed=copy.deepcopy(original);changed['status']='running'
        with self.assertRaises(ValueError):aligned(original,changed)

    def test_public_export_omits_raw_text_and_exception_details(self):
        report={'status':'complete','results':[{'id':'case','sha256':'hash','tagged':'secret',
            'text':'secret','error':'Exception containing secret','detected':None}]}
        exported=public_report(report)
        self.assertEqual(exported['results'],[{'id':'case','sha256':'hash','detected':None,'error':'inspection_failed'}])
        self.assertNotIn('secret',str(exported))
        with self.assertRaises(ValueError):public_report({'status':'running'})

    def test_encoder_predictions_must_match_source_offsets(self):
        match={'start':3,'end':7,'text':'word'}
        self.assertEqual(source_spans('a: word',['password'],{'entities':{'password':[match]}}),[(3,7,'password')])
        for invalid in ({**match,'text':'invented'},{**match,'end':8},{**match,'start':7}):
            with self.assertRaises(ValueError):source_spans('a: word',['password'],{'entities':{'password':[invalid]}})
        with self.assertRaises(ValueError):source_spans('a: word',['password'],{'entities':{'unrequested':[match]}})


if __name__=='__main__':unittest.main()
