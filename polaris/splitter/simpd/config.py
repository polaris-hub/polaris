from typing import List
from typing import Optional
from typing import Tuple


from pydantic import BaseModel


class SIMPDConfig(BaseModel):
    target_train_frac_active: float = -1
    target_test_frac_active: float = -1
    target_delta_test_frac_active: Optional[float] = None
    target_GF_delta_window: Tuple[int, int] = (10, 30)
    target_G_val: int = 70
    max_population_cluster_entropy: float = 0.9
