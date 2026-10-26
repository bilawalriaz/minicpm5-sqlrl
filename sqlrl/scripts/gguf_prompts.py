# Usage: python scripts/gguf_prompts.py <llama-server port> <label>
# Runs the LM Studio example prompts against a llama-server at temperature 0, executing run_sql
# on eval.db and stopping at submit_sql. Prints whether the submitted result matches the expected answer.
import json, re, sqlite3, sys, urllib.request
port, label = sys.argv[1], sys.argv[2]
root = '/Users/bilawalriaz/minicpm5-sqlrl'
system = open(f'{root}/docs/lmstudio-system-prompt.txt').read()
text = open(f'{root}/docs/lmstudio-prompts.txt').read()
items = re.findall(r'^\d+\.\s*(.+?)\n\s+Expected:\s*(.+?)\s{2}\(', text, re.M)
db = sqlite3.connect(f'file:{root}/sqlrl/data/sql-v2.2/eval.db?mode=ro', uri=True)
call = re.compile(r'<function name="(run_sql|submit_sql)"><param name="query">(.*?)</param></function>', re.S)
def chat(msgs):
    body = json.dumps({'messages': msgs, 'temperature': 0, 'top_p': 1, 'min_p': 0, 'repeat_penalty': 1.0, 'max_tokens': 384, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'}), timeout=300)
    return json.load(r)['choices'][0]['message']['content']
def run(sql):
    try:
        rows = db.execute(sql).fetchmany(10001)
        return rows, '\n'.join(' | '.join(str(v) for v in r) for r in rows[:10]) or '(no rows)'
    except Exception as e:
        return None, f'error: {e}'
ok = 0
for i, (q, expected) in enumerate(items, 1):
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': q}]
    verdict, turns = 'no submit', 0
    for turns in range(1, 6):
        out = chat(msgs)
        m = call.search(out)
        if not m: verdict = 'no tool call'; break
        rows, shown = run(m.group(2))
        if m.group(1) == 'submit_sql':
            got = '; '.join(' | '.join(str(v) for v in r) for r in rows) if rows is not None else shown
            verdict = 'correct' if rows is not None and sorted(got.split('; ')) == sorted(expected.split('; ')) else f'wrong ({got[:60]})'
            break
        msgs += [{'role': 'assistant', 'content': out}, {'role': 'user', 'content': f'<tool_response>\n{shown}\n</tool_response>'}]
    ok += verdict == 'correct'
    print(f'{label} {i:2d} {verdict} turns={turns}', flush=True)
print(f'{label} total {ok}/{len(items)}')
