"""agenthatch 异常层次。

v0.1 仅定义两个基础异常。
后续版本按需扩展（SkillParseError → v0.2, AgentNotFoundError → v0.4）。
"""


class AgentHatchError(Exception):
    """agenthatch 基础异常。"""
    exit_code = 1


class ConfigError(AgentHatchError):
    """配置文件错误。"""
    exit_code = 2
