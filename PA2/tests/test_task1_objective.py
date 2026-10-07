"""Mathematical objective and masking checks; no public weights downloaded."""
import math
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from common.metrics import sampled_kl
from common.data import pad_batch
from task1_dpo.dpo import dpo_loss
from task1_dpo.utils import sequence_logprobs,token_logprobs


class ObjectiveTests(unittest.TestCase):
    def test_reference_cancels_and_gradients_prefer_chosen(self):
        pc=torch.tensor([-3.,-7.],requires_grad=True);pr=torch.tensor([-6.,-2.],requires_grad=True)
        rc=pc.detach().clone().requires_grad_();rr=pr.detach().clone().requires_grad_()
        loss,d=dpo_loss(pc,pr,rc,rr,0.1)
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        self.assertEqual(d['preference_accuracy'].item(),0)
        loss.backward()
        self.assertTrue((pc.grad<0).all());self.assertTrue((pr.grad>0).all())
        self.assertIsNone(rc.grad);self.assertIsNone(rr.grad)

    def test_reference_preference_is_not_policy_improvement(self):
        loss,d=dpo_loss(torch.tensor([3.]),torch.tensor([0.]),torch.tensor([2.]),torch.tensor([0.]),.3)
        self.assertAlmostEqual(d['logit_mean'].item(),.3,places=6)
        self.assertAlmostEqual(loss.item(),-math.log(1/(1+math.exp(-.3))),places=6)

    def test_chunked_logp_and_gradient_match_logsoftmax(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.table=torch.nn.Parameter(torch.randn(2,5,11))
            def forward(self,**kw):return SimpleNamespace(logits=self.table)
        model=Model();ids=torch.tensor([[0,2,3,4,5],[0,0,6,7,8]])
        mask=torch.tensor([[0.,0,1,1,1],[0.,0,0,1,1]])
        batch={'input_ids':ids,'attention_mask':(ids!=0).long(),'response_mask':mask}
        seq,tok,valid=sequence_logprobs(model,batch)
        expected=F.log_softmax(model.table[:,:-1].float(),-1).gather(-1,ids[:,1:,None]).squeeze(-1)
        torch.testing.assert_close(tok,expected)
        torch.testing.assert_close(seq,(expected*mask[:,1:]).sum(-1))
        seq.sum().backward();got=model.table.grad.clone();model.table.grad=None
        (expected*mask[:,1:]).sum().backward();torch.testing.assert_close(got,model.table.grad)
        self.assertEqual(got[0,0].abs().sum().item(),0) # Prompt target excluded.
        self.assertEqual(got[:,4].abs().sum().item(),0) # No next target at final input.

    def test_sampled_kl_uses_token_weighting(self):
        pc=torch.tensor([[2.,2.,0.],[4.,0.,0.]]);ref=torch.zeros_like(pc)
        mask=torch.tensor([[1.,1.,0.],[1.,0.,0.]])
        self.assertAlmostEqual(sampled_kl(pc,ref,mask).item(),8/3,places=6)

if __name__=='__main__':unittest.main()
