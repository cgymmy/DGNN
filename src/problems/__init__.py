from problems.poisson1d_dg import Poisson1D_dg
from problems.poisson1d_base import Poisson1D_base
from problems.poisson2d_dg import Poisson2d_dg
from problems.poisson2d_base import Poisson2d_base
from problems.burgers_dg import Burgers_dg
from problems.burgers_pinn import Burgers_pinn
from problems.burgers_hpvpinn import Burgers_hpVPINN

__all__ = [
    'Poisson1D_dg', 'Poisson1D_base',
    'Poisson2d_dg', 'Poisson2d_base',
    'Burgers_dg', 'Burgers_pinn', 'Burgers_hpVPINN',
]
