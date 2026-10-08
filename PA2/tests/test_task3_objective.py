import unittest
import torch
from task3_grpo.grpo import group_relative_advantages,grpo_policy_loss,mask_truncated_sequences
from task3_grpo.analyze_groups import analyze_cache,validate_cache


class GRPOObjectiveTests(unittest.TestCase):
    def test_prompt_independence_and_constant_groups(self):
        rewards=torch.tensor([1.,3.,100.,104.,7.,7.]);groups=torch.tensor([0,0,1,1,2,2])
        a=group_relative_advantages(rewards,groups)
        torch.testing.assert_close(a,torch.tensor([-1.,1.,-1.,1.,0.,0.]),atol=2e-6,rtol=0)
        shifted=rewards.clone();shifted[groups==1]+=1000
        torch.testing.assert_close(a,group_relative_advantages(shifted,groups))
        with self.assertRaises(ValueError):group_relative_advantages(rewards,groups[:2])

    def test_normalization_length_gradient_and_truncation(self):
        mask=torch.tensor([[1.,1.,0.,0.],[1.,1.,1.,1.],[1.,1.,1.,1.]])
        mask=mask_truncated_sequences(mask,[False,False,True]);adv=torch.tensor([1.,1.,10.])
        old=torch.zeros(3,4)
        gradients={}
        for kind in ('grpo','dr_grpo'):
            new=old.clone().requires_grad_()
            loss,_=grpo_policy_loss(new,old,adv,mask,old,.2,0.,kind,4);loss.backward();gradients[kind]=new.grad
            self.assertTrue((new.grad[2]==0).all())
        self.assertAlmostEqual(float(gradients['grpo'][0,0]/gradients['grpo'][1,0]),2.)
        self.assertAlmostEqual(float(gradients['dr_grpo'][0,0]/gradients['dr_grpo'][1,0]),1.)
        self.assertAlmostEqual(float(gradients['grpo'][0].sum()),float(gradients['grpo'][1].sum()))

    def test_clipping_uses_minimum_for_both_signs(self):
        old=torch.zeros(2,1);new=torch.tensor([[.5],[-.5]],requires_grad=True)
        loss,_=grpo_policy_loss(new,old,torch.tensor([1.,-1.]),torch.ones(2,1),old,.2,0.)
        loss.backward();torch.testing.assert_close(new.grad,torch.zeros_like(new))
        self.assertAlmostEqual(float(loss.detach()),-.2,places=6)

    def test_cache_equal_generations_fixed_bins_and_all_constant(self):
        rows=[{'prompt_id':str(p),'source_index':p,'generation_index':i,'reward':float(p+i%2),
               'clipped_at_max':False} for p in range(4) for i in range(8)]
        result=analyze_cache(rows)
        for bin_name,n in [('all',32),('lower_reward',16),('higher_reward',16)]:
            summaries=[r for r in result['summary'] if r['difficulty_bin']==bin_name]
            self.assertEqual([r['total_generations'] for r in summaries],[n,n,n])
        assignment={r['prompt_id']:r['difficulty_bin'] for r in result['difficulty_assignments']}
        self.assertTrue(all(r['difficulty_bin']==assignment[r['prompt_id']] for r in result['groups']))
        for row in rows:row['reward']=1.
        constant=analyze_cache(rows)
        self.assertTrue(all(r['informative_group_rate']==0 for r in constant['summary']))
        with self.assertRaises(ValueError):validate_cache(rows[:-1])
        with self.assertRaises(ValueError):analyze_cache(rows,[3])
if __name__=='__main__':unittest.main()
