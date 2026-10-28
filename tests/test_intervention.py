import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
torch = pytest.importorskip("torch")   # legacy PyTorch implementation
from torch import nn
from intervention_campaign import forced_linear_error, final_rollout_error, local_jacobian_penalty

class Poly(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Parameter(torch.tensor(0.9, dtype=torch.float64))
    def forward(self,x):
        return self.a*x + 0.02*x*x


def test_one_step_exact_defect_and_grads():
    m=Poly()
    w=torch.tensor([[[1.,2.],[1.1,2.1]]],dtype=torch.float64)
    d=forced_linear_error(m,w)
    assert torch.allclose(d, m(w[:,0])-w[:,1], atol=1e-12)
    d.square().mean().backward()
    assert m.a.grad is not None and torch.isfinite(m.a.grad)


def test_linear_model_matches_exact_rollout_error_all_horizons():
    for h in [1,2,4,8]:
        m=nn.Linear(3,3, dtype=torch.float64)
        w=torch.randn(2,h+1,3,dtype=torch.float64)
        assert torch.allclose(forced_linear_error(m,w), final_rollout_error(m,w), atol=2e-11)


def test_jvp_gradient_matches_finite_difference():
    m=Poly()
    w=torch.tensor([[[0.5],[0.6],[0.7],[0.8]]],dtype=torch.float64)
    loss=forced_linear_error(m,w).square().mean()
    (gr,)=torch.autograd.grad(loss,m.a)
    orig=m.a.item()
    eps=1e-5
    with torch.no_grad():m.a.fill_(orig+eps)
    plus=forced_linear_error(m,w).square().mean().item()
    with torch.no_grad():m.a.fill_(orig-eps)
    minus=forced_linear_error(m,w).square().mean().item()
    with torch.no_grad():m.a.fill_(orig)
    fd=(plus-minus)/(2*eps)
    assert abs(gr.item()-fd) < 1e-6


def test_fno_jvp_supports_backward():
    import ks_fno_campaign as ks
    torch.manual_seed(4)
    m=ks.FNO1d(16,modes=4,width=4,depth=2,projection_width=8,dtype=torch.float64)
    w=torch.randn(1,3,16,dtype=torch.float64)
    loss=forced_linear_error(m,w).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_defect_energy_one_step_equals_one_step_mse():
    from intervention_campaign import propagated_defect_energy
    m=Poly()
    w=torch.tensor([[[1.,2.],[1.1,2.1]]],dtype=torch.float64)
    expected=(m(w[:,0])-w[:,1]).square().mean()
    got=propagated_defect_energy(m,w)
    assert torch.allclose(got, expected, atol=1e-12)


def test_defect_energy_differs_from_vector_sum_when_contributions_cancel():
    from intervention_campaign import propagated_defect_energy, forced_linear_error
    # Linear identity model with teacher-forced targets chosen so e0=+1, e1=-1.
    m=nn.Linear(1,1,bias=False,dtype=torch.float64)
    with torch.no_grad(): m.weight.fill_(1.0)
    w=torch.tensor([[[0.0],[-1.0],[0.0]]],dtype=torch.float64)
    vec=forced_linear_error(m,w).square().mean()
    ene=propagated_defect_energy(m,w)
    assert vec.item() == 0.0
    assert ene.item() > 0.0


def test_defect_energy_fno_supports_backward():
    from intervention_campaign import propagated_defect_energy
    import ks_fno_campaign as ks
    torch.manual_seed(5)
    m=ks.FNO1d(16,modes=4,width=4,depth=2,projection_width=8,dtype=torch.float64)
    w=torch.randn(1,4,16,dtype=torch.float64)
    loss=propagated_defect_energy(m,w)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
