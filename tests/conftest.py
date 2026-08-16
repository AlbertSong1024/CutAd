# -*- coding: utf-8 -*-
"""确保测试运行时项目根目录在 sys.path 中（便于导入 fuckad 包）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
