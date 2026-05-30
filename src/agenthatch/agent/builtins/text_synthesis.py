"""Text synthesis builtin capability."""

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability


class TextSynthesisCap(BuiltinCapability):
    name = "text_synthesis"
    cap_type = "reasoning"
    description = "Synthesize and summarize text content"
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to synthesize"},
            "format": {
                "type": "string",
                "description": "Output format hint (summary/bullets/paragraph)",
            },
        },
        "required": ["text"],
    }

    def execute(self, text: str = "", format: str = "summary") -> str:
        lines = [line for line in text.split("\n") if line.strip()]
        word_count = len(text.split())
        if format == "bullets":
            items = [f"- {line[:120]}" for line in lines[:20]]
            return "\n".join(items) if items else text[:500]
        elif format == "paragraph":
            return " ".join(text.split()[:200])
        else:
            return f"[{word_count} words, {len(lines)} lines]\n{text[:2000]}"


BUILTIN_REGISTRY["text_synthesis"] = TextSynthesisCap
