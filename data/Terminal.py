# -*- coding: utf-8 -*-
"""
Waip AI Q-DRN · ChatUI 工作区终端
================================
流程：
  1. 启动 → 选择或新建意识体，并加载三个数据库；
  2. 打开固定 390×844 的 Python ChatUI 窗口；
  3. 每轮读取 390×192 视野灰度像素，加上视野和光标的归一化位置，写入 74,884 个输入；
  4. 输出 0/1 控制光标 X/Y，输出 2 控制视野 Y，输出 3 大于 0 时点击一次；
  5. 关闭窗口时保存数据库；&& create 新建 74,884 输入 / 4 输出的最终配置意识体。

运行：python data/Terminal.py
"""

import importlib.util
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent          # data/
ROOT_DIR = DATA_DIR.parent                          # 项目根目录
AGENT_DIR = DATA_DIR / "Agent"
MAIN_FILE = ROOT_DIR / "main" / "main.py"
DATA_INTO_FILE = ROOT_DIR / "main" / "data_into.py"

# 三个数据库文件名
NEURAL_CONFIG_DB = "Neural Configuration Database.json"
SYNAPSES_DB = "Synapses Database.json"
NEURAL_STATE_DB = "Neural State Database.json"

EXIT_COMMAND = "&& exit"        # 唯一的退出命令
CREATE_COMMAND = "&& create"    # 新建意识体的命令


def is_exit(text):
    """只有输入 && exit 才算退出命令"""
    return " ".join(text.split()).lower() == EXIT_COMMAND


def is_create(text):
    """只有输入 && create 才算新建意识体命令"""
    return " ".join(text.split()).lower() == CREATE_COMMAND


def clean(text):
    """去掉首尾空白与可能的 BOM（Windows 管道输入会出现）"""
    return text.replace("\ufeff", "").strip()


def load_main_func():
    """按文件路径加载 main/main.py 中的 main 函数"""
    spec = importlib.util.spec_from_file_location("drn_main", MAIN_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {MAIN_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def load_data_into_func():
    """按文件路径加载 main/data_into.py 中的 data_into 函数"""
    spec = importlib.util.spec_from_file_location("drn_data_into", DATA_INTO_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {DATA_INTO_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.data_into


def list_agents():
    """列出 data/Agent 下所有 Agent 目录"""
    if not AGENT_DIR.exists():
        return []
    return sorted(p.name for p in AGENT_DIR.iterdir() if p.is_dir())


def load_database(path):
    """读取一个 JSON 数据库；文件不存在或为空时返回空字典"""
    if not path.exists():
        print(f"  [警告] 缺少数据库文件：{path.name}，按空库处理")
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}
    except json.JSONDecodeError as e:
        print(f"  [警告] {path.name} 不是合法 JSON（{e}），按空库处理")
        return {}


def load_agent_databases(agent_name):
    """加载指定 Agent 的三个数据库"""
    agent_dir = AGENT_DIR / agent_name
    config_db = load_database(agent_dir / NEURAL_CONFIG_DB)
    synapses_db = load_database(agent_dir / SYNAPSES_DB)
    state_db = load_database(agent_dir / NEURAL_STATE_DB)
    print(f"  [完成] {NEURAL_CONFIG_DB} (配置项 {len(config_db)})")
    print(f"  [完成] {SYNAPSES_DB} (突触 {len(synapses_db)})")
    print(f"  [完成] {NEURAL_STATE_DB} (状态项 {len(state_db)})")
    return config_db, synapses_db, state_db


def ask_agent_name():
    """询问新意识体名称；返回合法且不重名的名称，取消时返回 None"""
    illegal = set('\\/:*?"<>|')
    while True:
        try:
            raw = clean(input("请输入新意识体名称（输入 && exit 取消）："))
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if is_exit(raw):
            return None
        if not raw:
            print("名称不能为空，请重新输入。")
            continue
        if any(c in illegal for c in raw):
            print("名称不能包含路径分隔符或 Windows 非法字符，请重新输入。")
            continue
        if (AGENT_DIR / raw).exists():
            print(f"意识体「{raw}」已存在，请换一个名称。")
            continue
        return raw


def write_database(path, data):
    """把字典写成 UTF-8 的 JSON 文件"""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def save_databases(agent_name, config_db, state_db, synapses_db):
    """把三个数据库写回该意识体的数据库文件"""
    agent_dir = AGENT_DIR / agent_name
    write_database(agent_dir / NEURAL_CONFIG_DB, config_db)
    write_database(agent_dir / NEURAL_STATE_DB, state_db)
    write_database(agent_dir / SYNAPSES_DB, synapses_db)


def create_agent():
    """新建最终配置意识体：74,884 输入、4 输出、500 记忆、2,000 正常神经元。"""
    print(f"\n[新建] 输入 && create 进入意识体创建流程")
    name = ask_agent_name()
    if name is None:
        print("已取消创建。")
        return None

    print(
        f"[新建] 正在初始化意识体「{name}」的数据库"
        "（74,884 输入 / 4 输出 / 500 记忆 / 2,000 正常）…"
    )
    try:
        data_into = load_data_into_func()
        config_db, synapses_db, state_db = data_into(
            input=74884, output=4, memory=500, Normal=2000
        )
    except Exception as e:
        print(f"[错误] 数据库初始化失败：{type(e).__name__}: {e}")
        return None

    agent_dir = AGENT_DIR / name
    try:
        agent_dir.mkdir(parents=True, exist_ok=True)
        write_database(agent_dir / NEURAL_CONFIG_DB, config_db)
        write_database(agent_dir / SYNAPSES_DB, synapses_db)
        write_database(agent_dir / NEURAL_STATE_DB, state_db)
    except OSError as e:
        print(f"[错误] 数据库写入失败：{e}")
        return None

    print(f"  [完成] 意识体「{name}」创建成功")
    print(f"  [完成] {NEURAL_CONFIG_DB} (神经元 {len(config_db)})")
    print(f"  [完成] {SYNAPSES_DB} (突触 {len(synapses_db)})")
    print(f"  [完成] {NEURAL_STATE_DB} (状态项 {len(state_db)})")
    return name


def choose_agent():
    """让用户选择 Agent，返回 Agent 名称；用户放弃选择时返回 None"""
    agents = list_agents()
    if agents:
        print("可用意识体：")
        for i, name in enumerate(agents, 1):
            print(f"  {i}. {name}")
    else:
        print(f"{AGENT_DIR} 下还没有任何意识体，可用 && create 新建。")

    while True:
        try:
            raw = clean(input("\n请输入要调用的意识体（序号或名称；&& create 新建；&& exit 退出）："))
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if is_exit(raw):
            return None
        if is_create(raw):
            created = create_agent()
            if created is not None:
                return created
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(agents):
            return agents[int(raw) - 1]
        if raw in agents:
            return raw
        print(f"未找到意识体「{raw}」，请重新输入。")


def main():
    print("=" * 56)
    print("Waip AI Q-DRN · 控制台")
    print("版本：2.6.3.0")
    print("开发者：科文通")
    print("=" * 56)
    print("启动后将弹出控制台：选择「训练」开始 GA 训练，或「运行模型」观看已有网络。\n")
    try:
        from Console import launch
    except Exception as e:
        print(f"[错误] 控制台加载失败：{type(e).__name__}: {e}")
        return 1
    return launch()


if __name__ == "__main__":
    sys.exit(main())
