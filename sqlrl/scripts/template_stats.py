# Usage: python scripts/template_stats.py [templates.db]
# Counts generated templates by status and groups rejection reasons into the categories used in the write-up.
import re, sqlite3, sys, collections
rules = [
 ('Reused a template name', r'^name is taken'),
 ('Empty or NULL answer', r'^empty or NULL answer'),
 ('Solver could not answer it', r'^fresh solver|solver'),
 ('Reviewer rejected it', r'^reviewer|review'),
 ('Same answer on every database', r'^same answer on every database'),
 ('Too many parameter combinations', r'parameter combinations'),
 ('Placeholders did not match', r'^placeholders differ'),
 ('A filter never changes the answer', r'^removing '),
 ('Same as an evaluation question', r'^same results as an evaluation question'),
 ('Duplicate of an accepted template', r'^same SQL as an accepted'),
 ('Reference SQL fails', r'^reference SQL fails'),
 ('Depends on tie order', r'tie order'),
 ('Overlaps a held-out question', r'held-out'),
]
db = sqlite3.connect(sys.argv[1] if len(sys.argv) > 1 else "data/sql-v2.2/templates.db")
rows = [r[0] for r in db.execute("select reason from templates where status='rejected'")]
c = collections.Counter()
for r in rows:
    c[next((n for n, p in rules if re.search(p, r)), 'Malformed template')] += 1
acc = db.execute("select count(*) from templates where status='accepted'").fetchone()[0]
print(len(rows), acc)
for n, v in c.most_common(): print(v, n)
