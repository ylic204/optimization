"""Streaming diagnostics; never repair a generated optimization model."""
import json
from pathlib import Path
from uuid import uuid4


class AlgebraOutputGuard:
    """Enforce the compiler's 1024-element array limit, including partial JSON.

    Repeated values are legal. Count array entries, not repeated tokens; quoted
    expressions are ignored. No template-specific constraints are inserted.
    """
    def __init__(self):
        self.error = None

    def __call__(self, text):
        stack = []
        quoted = escaped = False
        for char in text:
            if quoted:
                if escaped: escaped = False
                elif char == '\\': escaped = True
                elif char == '"': quoted = False
                continue
            if char.isspace(): continue
            if stack and stack[-1][0] == '[' and stack[-1][2] and char != ']':
                stack[-1][1] += 1
                stack[-1][2] = False
                if stack[-1][1] > 1024:
                    self.error = 'JSON array exceeds compiler limit of 1024 entries'
                    return 'array_limit'
            if char == '"': quoted = True
            elif char in '[{': stack.append([char, 0, True])
            elif char in ']}':
                if stack: stack.pop()
            elif char == ',' and stack and stack[-1][0] == '[':
                stack[-1][2] = True
        try:
            if isinstance(json.loads(text), dict): return 'json_complete'
        except ValueError: pass
        return None


class GenerationTrace:
    def __init__(self, directory, sample_id, enabled=True):
        self.enabled = enabled
        self.printed = ''
        self.path = None
        if directory:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / (uuid4().hex + '.txt')
            self.path.write_text('', encoding='utf-8')
        if enabled:
            print(f'\n[Qwen start] sample={sample_id} raw_answer={self.path}', flush=True)

    def __call__(self, text):
        if self.path: self.path.write_text(text, encoding='utf-8')
        # A final incomplete unicode token can decode differently next step.
        stable = text.rstrip('\ufffd')
        if self.enabled and stable.startswith(self.printed):
            print(stable[len(self.printed):], end='', flush=True)
            self.printed = stable

    def finish(self, text, reason):
        self(text)
        if self.enabled: print(f'\n[Qwen finish] reason={reason}', flush=True)
