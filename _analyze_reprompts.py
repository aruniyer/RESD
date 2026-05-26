import json, random
random.seed(42)
with open('/data/ariy/RESD/checkpoints/codistill_repro_math/qwen25math_1.5b_sdpo_phase2/reprompts/1.jsonl') as f:
    lines = f.readlines()
samples = [json.loads(l) for l in lines]
incorrect = [s for s in samples if s['score'] == 0.0]
print(f'Total samples: {len(samples)}, Incorrect (getting distillation): {len(incorrect)}')
print()
for s in random.sample(incorrect, min(3, len(incorrect))):
    reprompt = s['reprompt']
    if 'Correct solution:' in reprompt:
        sol_start = reprompt.index('Correct solution:')
        sol_end = reprompt.index('Correctly solve the original question.')
        solution = reprompt[sol_start:sol_end].strip()
    else:
        solution = '(no solution found)'
    print('='*60)
    print(f'QUESTION: ...{s["prompt"][-150:]}')
    print(f'STUDENT (wrong): {s["response"][:200]}')
    print(f'DEMO SOLUTION: {solution[:400]}')
    print()
