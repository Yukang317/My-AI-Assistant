"""已弃用：主入口已更名为 app.py。本文件仅保留旧启动命令兼容。

请改用：uv run app.py
"""

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    print("[提示] step23_rag.py 已更名为 app.py，建议改用: uv run app.py")
    runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
