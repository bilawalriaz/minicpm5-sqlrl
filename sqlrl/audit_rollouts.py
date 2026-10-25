"""Recheck archived SQL against original data plus the v2 verifier fixtures.

This compares SQL semantics, not compatibility with the v2 submission protocol.
Usage: python audit_rollouts.py --run curriculum_50b --out engineering/legacy-sql-audit.json
"""
import argparse
import json
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path

import env


def audit(run, output):
    root = Path(__file__).parent
    counts, examples, cache, changed_templates = Counter(), [], {}, Counter()
    with ExitStack() as stack:
        connections = [stack.enter_context(closing(c)) for c in env.verifier_connections()]
        legacy_paths = [root / 'shop.db', root / 'shop_alt1.db', root / 'shop_alt2.db']
        for legacy in reversed(legacy_paths):
            if legacy.exists():
                connections.insert(0, stack.enter_context(closing(env.connect(legacy))))
        files = sorted((root / 'runs' / run / 'samples').glob('step_*.json'))
        for path in files:
            for record in json.loads(path.read_text()):
                if not record.get('final_sql'):
                    counts['no_recorded_submission'] += 1
                    continue
                key = (record['final_sql'], record['gold_sql'])
                if key not in cache:
                    cache[key] = env.grade(connections, *key)
                ok, reason = cache[key]
                old_ok = record['outcome'] == 'correct'
                label = f"{'accepted' if old_ok else 'rejected'}_to_{'accepted' if ok else 'rejected'}"
                counts[label] += 1
                if old_ok and not ok:
                    changed_templates[record['template']] += 1
                if old_ok and not ok and len(examples) < 12:
                    examples.append({'step': path.stem, 'template': record['template'],
                                     'source': record.get('source', 'student'),
                                     'question': record['question'], 'sql': key[0],
                                     'reference': key[1], 'reason': reason})
    result = {'run': run, 'verifier_version': env.VERIFIER_VERSION,
              'comparison': 'SQL semantics on original data plus v2 fixtures; legacy protocol not regraded',
              'steps': len(files), 'counts': dict(counts), 'changed_templates': dict(changed_templates), 'examples': examples}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.out)
