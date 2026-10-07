"""Verify context equality, response priority, masks, and extreme overflow."""
import unittest
from tests.test_task1_pipeline import TinyTokenizer
from task1_dpo.encoding import encode_preference_pair


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tok = TinyTokenizer()
        self.messages = [{'role':'user','content':' '.join('context'+str(i) for i in range(20))}]

    def test_shared_prompt_and_intact_responses(self):
        c, r, d = encode_preference_pair(self.tok,self.messages,'one two three','one',10)
        cp = [i for i,m in zip(*c) if not m]
        rp = [i for i,m in zip(*r) if not m]
        self.assertEqual(cp,rp)
        self.assertEqual(cp,self.tok.apply_chat_template(self.messages,tokenize=True,add_generation_prompt=True)[-6:])
        self.assertEqual(c[0][-4:],self.tok.encode('one two three')+[2])
        self.assertEqual(r[0][-2:],self.tok.encode('one')+[2])
        self.assertEqual(d['chosen_tokens_removed'],0)
        self.assertEqual(d['rejected_tokens_removed'],0)
        self.assertGreater(d['prompt_tokens_removed'],0)
        self.assertEqual(len(c[0]),10)

    def test_extreme_response_retains_prompt_and_eos(self):
        text = ' '.join('response'+str(i) for i in range(20))
        c, r, d = encode_preference_pair(self.tok,self.messages,text,'short',8)
        self.assertEqual(c[1],[0]+[1]*7)
        self.assertEqual(r[0][0],c[0][0])
        self.assertEqual(c[0][-1],2)
        self.assertEqual(c[0][1:],(self.tok.encode(text)+[2])[-7:])
        self.assertGreater(d['chosen_tokens_removed'],0)
        self.assertEqual(d['rejected_tokens_removed'],0)

    def test_short_pair_unchanged(self):
        msgs=[{'role':'user','content':'question'}]
        c,r,d=encode_preference_pair(self.tok,msgs,'answer','other',64)
        prompt=self.tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
        self.assertEqual(c[0],prompt+self.tok.encode('answer')+[2])
        self.assertFalse(d['pair_overlength'])
        self.assertEqual(d['prompt_tokens_removed'],0)

if __name__ == '__main__': unittest.main()
