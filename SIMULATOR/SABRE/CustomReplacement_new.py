# CustomReplacement.py
# python sab.py --plugin CustomReplacement.py --replace-plugin custom_replacement ...

try:
    from __main__ import ReplaceStrategy, AbrContext, register_replace
except ImportError:
    from sab import ReplaceStrategy, AbrContext, register_replace


@register_replace("custom_replacement")
class CustomReplacement(ReplaceStrategy):
    def __init__(self):
        self.replacing = None

    def check_replace(self, ctx: AbrContext, quality: int):
        self.replacing = None
        buf = ctx.buffer.contents
        for i in range(2, len(buf)):
            if buf[i] < quality:
                self.replacing = i - len(buf)
                return self.replacing
        return None

    def check_abandon(self, progress, buffer_level_ms: float):
        return None
