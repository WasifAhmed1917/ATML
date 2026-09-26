# Notebook-derived stage: ATML_PA1_Task2.ipynb
# Run through common.stages.Experiment, not as an independent module.

# %% Original notebook cell 8
def mmd_rbf(a,b):
    combined=torch.cat([a,b],0)
    sq=torch.cdist(combined,combined,p=2).square()
    with torch.no_grad():
        upper=sq.triu(diagonal=1)
        values=upper[upper>0]
        median=values.median().clamp_min(1e-8) if len(values) else sq.new_tensor(1.)
    kernel=sum(torch.exp(-sq/(2*scale*median)) for scale in (0.5,1.,2.))
    n=len(a)
    return kernel[:n,:n].mean()+kernel[n:,n:].mean()-2*kernel[:n,n:].mean()
class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,alpha):ctx.alpha=alpha;return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return -ctx.alpha*g,None
class Discriminator(nn.Sequential):
    def __init__(self,dim):super().__init__(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,2))
def domain_input(method,feat,logits):
    feat=F.normalize(feat,p=2,dim=1,eps=1e-6)*math.sqrt(feat.shape[1])
    if method=='CDAN':
        p=logits.softmax(1)
        return torch.bmm(p.unsqueeze(2),feat.unsqueeze(1)).flatten(1)
    return feat
f=torch.randn(4,512,requires_grad=True); q=torch.randn(4,7,requires_grad=True)
domain_input('CDAN',f,q).sum().backward()
assert f.grad is not None and q.grad is not None
assert torch.isfinite(mmd_rbf(torch.randn(24,512),torch.randn(24,512)))
