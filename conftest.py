import pathlib
import sys

# 扁平布局：包就在仓库根目录。显式加进 sys.path，不再依赖全局 PYTHONPATH。
_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
