"""Environment packages implemented in the routing repository."""

from pathlib import Path

_skillrl_env_package_path = (
    Path(__file__).resolve().parents[3]
    / "SkillRL"
    / "agent_system"
    / "environments"
    / "env_package"
)
if _skillrl_env_package_path.exists():
    __path__.append(str(_skillrl_env_package_path))
