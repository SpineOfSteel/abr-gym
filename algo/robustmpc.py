# robustmpc.py
try:
    from __main__ import register_abr, AbrContext
except ImportError:
    from sab import register_abr, AbrContext

#FIX ME
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
    
from shim_base import HttpShimBase


@register_abr("robustmpc")
class RobustMpcAbr(HttpShimBase):
    SHIM_NAME = "robustmpc"

    def __init__(self, cfg: dict, ctx: AbrContext):
        super().__init__(cfg, ctx)
        if cfg.get("shim"):
            self.server_url = f'http://127.0.0.1:{cfg.get("shim")}/'
            print('URL: ',self.server_url)

