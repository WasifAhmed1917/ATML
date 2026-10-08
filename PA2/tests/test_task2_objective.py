import unittest
import torch
from task2_ppo.ppo import compute_gae,shaped_rewards,ppo_policy_loss,normalize_advantages,active_clip_fraction

class PPOObjectiveTests(unittest.TestCase):
    def test_clipping_both_signs_and_padding(self):
        ratios=torch.tensor([[1.5,.5,.5,1.5,100.]])
        a=torch.tensor([[1.,-1.,1.,-1.,999.]])
        mask=torch.tensor([[1.,1.,1.,1.,0.]])
        loss,r,clip=ppo_policy_loss(ratios.log(),torch.zeros_like(ratios),a,mask,.2)
        self.assertAlmostEqual(float(loss),.15,places=6)
        self.assertEqual(float(clip),1.)
        self.assertEqual(float(active_clip_fraction(r,a,mask,.2)),.5)
        x=ratios.log().requires_grad_();ppo_policy_loss(x,torch.zeros_like(x),a,mask,.2)[0].backward()
        self.assertEqual(float(x.grad[0,0]),0.);self.assertEqual(float(x.grad[0,1]),0.)
        self.assertNotEqual(float(x.grad[0,2]),0.);self.assertNotEqual(float(x.grad[0,3]),0.)
        self.assertEqual(float(x.grad[0,4]),0.)

    def test_terminal_reward_and_variable_length_gae(self):
        mask=torch.tensor([[1.,1.,0.],[1.,1.,1.]])
        old=torch.ones(2,3);ref=torch.zeros(2,3)
        r=shaped_rewards(torch.tensor([2.,3.]),old,ref,mask,.1)
        torch.testing.assert_close(r,torch.tensor([[-.1,1.9,0.],[-.1,-.1,2.9]]))
        rewards=torch.tensor([[1.,2.,0.]])
        values=torch.tensor([[.5,.25,99.]])
        a,returns=compute_gae(rewards,values,mask[:1],1.,1.)
        torch.testing.assert_close(a,torch.tensor([[2.5,1.75,0.]]))
        torch.testing.assert_close(returns[:,:2],torch.tensor([[3.,2.]]))

    def test_normalize_valid_only(self):
        mask=torch.tensor([[1.,1.,0.]])
        normalized=normalize_advantages(torch.tensor([[2.,4.,1000.]]),mask)
        torch.testing.assert_close(normalized,torch.tensor([[-1.,1.,0.]]))
if __name__=='__main__':unittest.main()
