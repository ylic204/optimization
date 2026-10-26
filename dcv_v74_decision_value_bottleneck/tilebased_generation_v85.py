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


def export_model_artifacts(trace, item, enabled=False):
    """Persist partial/final modeling stages even when validation/solving fails."""
    if trace is None or trace.path is None: return
    base=trace.path.with_suffix('')
    artifacts={}
    for key in ('patch_node_registry','generated_model','compiled_model','solver_result'):
        if key in item:
            path=base.with_name(base.name+'.'+key+'.json')
            path.write_text(json.dumps(item[key],indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
            artifacts[key]=str(path.resolve())
    raw=item.get('generated_model')
    if isinstance(raw,dict):
        lines=['QWEN-GENERATED MATHEMATICAL MODEL (unvalidated unless compilation succeeded)',
               'Sets: '+json.dumps(raw.get('sets'),ensure_ascii=False),
               'Parameters: '+json.dumps(raw.get('parameters'),ensure_ascii=False),
               'Variables: '+json.dumps(raw.get('variables'),ensure_ascii=False),
               'Objective: '+json.dumps(raw.get('objective'),ensure_ascii=False), 'Constraints:']
        for rule in raw.get('constraints',[]) if isinstance(raw.get('constraints'),list) else []:
            lines.append(json.dumps(rule,ensure_ascii=False))
        path=base.with_name(base.name+'.model.txt')
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        artifacts['readable_model']=str(path.resolve())
    item['model_artifacts']=artifacts
    report=base.with_name(base.name+'.status.json')
    report.write_text(json.dumps({k:item.get(k) for k in
        ('sample_id','status','failure_category','error','model_answer_file','model_artifacts','patch_nodes')},indent=2)+'\n',encoding='utf-8')
    item['model_status_file']=str(report.resolve())
    if enabled:
        if isinstance(raw,dict):
            print('[Generated model] objective='+json.dumps(raw.get('objective')),flush=True)
            print('[Generated model] variables='+json.dumps(raw.get('variables')),flush=True)
            print('[Generated model] constraints='+json.dumps(raw.get('constraints')),flush=True)
        if 'compiled_model' in item:
            m=item['compiled_model']
            print(f"[Compiled model] variables={len(m['variable_labels'])} rows={len(m['constraints'])}",flush=True)
        print('[Model files] '+json.dumps(artifacts),flush=True)
