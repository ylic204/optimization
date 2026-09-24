from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent
files=['train_gdfm_v7.py','train_outcome_rl_v7.py','policy.py','lazy_inference_v7.py','evaluate_v7.py']
text='\n'.join((ROOT/f).read_text() for f in files)
checks={
    'DPO_present': bool(re.search(r'\\bdpo\\b|pair_acc|preference_head',text,re.I)),
    'STOP_present': bool(re.search(r'stop_head|STOP action',text,re.I)),
    'critic_present': bool(re.search(r'value_head|critic|GAE',text)),
    'soft_high_feature_mix': bool(re.search(r'w\\s*\\*\\s*high_feat|w\\[.*None.*\\]\\s*\\*\\s*high_feat',text)),
    'preview_aux_classification': bool(re.search(r'preview.*cross_entropy|preview.*binary_cross_entropy',text,re.I)),
}
print('=== V7 architecture / leakage audit ===')
for k,v in checks.items(): print(k, 'FAIL' if v else 'PASS')
if any(checks.values()): raise SystemExit(2)
