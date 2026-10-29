"""训练与推理共用的 YAML 读取和路径规范化工具。"""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"


def load_yaml(path) -> dict:
    """读取配置文件并确认顶层为映射，不依赖训练器或模型。"""
    with Path(path).expanduser().resolve().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("配置文件的顶层必须是 YAML 映射")
    return config


def normalize_path(value, *, base_dir=PROJECT_ROOT):
    """展开用户目录并解析绝对路径，保留空值且不要求目标已经存在。

    YAML 相对路径使用项目根目录；命令行调用者应明确传入当前工作目录。
    """
    if value is None or value == "":
        return value
    target = Path(value).expanduser()
    if not target.is_absolute():
        target = Path(base_dir).expanduser().resolve() / target
    return str(target.resolve())


def normalize_training_paths(config: dict, *, base_dir=PROJECT_ROOT) -> dict:
    """就地规范化训练的数据、检查点输入与输出路径，并返回同一配置。

    字段是否必填由训练器校验；此工具只处理已存在的路径，避免相互导入。
    """
    for section, key in (("data", "root"), ("train", "snapshots_folder"),
                         ("train", "pretrain_dir"), ("validation", "low_dir"),
                         ("validation", "high_dir"), ("output", "root")):
        values = config.get(section)
        if isinstance(values, dict) and key in values:
            values[key] = normalize_path(values[key], base_dir=base_dir)
    datasets = config.get("validation", {}).get("datasets", {})
    if isinstance(datasets, dict):
        for values in datasets.values():
            if isinstance(values, dict):
                for key in ("low_dir", "high_dir"):
                    if key in values:
                        values[key] = normalize_path(values[key], base_dir=base_dir)
    return config
