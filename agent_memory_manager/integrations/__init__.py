from .langchain import AgentMemoryManagerAdapter


def __getattr__(name: str):
    if name == "LlamaIndexMemoryAdapter":
        from .llamaindex import LlamaIndexMemoryAdapter  # noqa: PLC0415
        return LlamaIndexMemoryAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentMemoryManagerAdapter", "LlamaIndexMemoryAdapter"]
