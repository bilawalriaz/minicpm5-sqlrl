from types import SimpleNamespace

import evolve


def spec():
    return {'name': 'city_counter_test', 'tier': 1, 'skills': ['filter', 'aggregate'],
            'target_skill': 'filter', 'params': {'city': ['London', 'Paris', 'Tokyo', 'Seattle']},
            'question': 'How many customers live in {city}?',
            'sql': "SELECT COUNT(*) FROM customers WHERE city='{city}'"}


def test_template_validates_distinct_questions_and_all_parameters():
    seen = []
    teacher = SimpleNamespace(solve=lambda task: seen.append(task.question) or {'correct': True})
    good = spec()
    ok, reason = evolve.validate(good, teacher, set())
    assert ok and 'all 4 combinations' in reason
    assert len(seen) == len(set(seen)) == 3
    # A broken reference is rejected before the teacher supplies training traces.
    good['sql'] = "SELECT missing FROM customers WHERE city='{city}'"
    ok, reason = evolve.validate(good, teacher, set())
    assert not ok and 'reference SQL fails' in reason


def test_malformed_template_and_parameter_explosion():
    bad = spec()
    bad['question'] = '{city'
    assert evolve.check_structure(bad) == 'malformed placeholders'
    bad = spec()
    bad['params']['city'] = [float('inf'), 'Paris']
    assert 'not plain' in evolve.check_structure(bad)


def test_template_validation_rejects_free_reward_and_eval_duplicates():
    teacher = SimpleNamespace(solve=lambda task: {'correct': True})
    cases = {
        "SELECT COUNT(*) FROM customers WHERE city='{city}' AND id < 0": 'same answer on every database',
        "SELECT name FROM customers WHERE city='{city}' AND id < 0": 'empty or NULL answer',
        "SELECT name FROM customers WHERE city='{city}' ORDER BY country LIMIT 1": 'tie order',
    }
    for sql, reason in cases.items():
        bad = spec()
        bad['sql'] = sql
        ok, why = evolve.validate(bad, teacher, set())
        assert not ok and reason in why, why
    # The same question as a held-out evaluation template, in other words.
    dup = spec()
    dup.update(question='Count orders from customers living in {country}.', params={'country': ['UK', 'France']},
               sql="SELECT COUNT(o.id) FROM customers c, orders o WHERE o.customer_id = c.id AND c.country = '{country}'")
    ok, why = evolve.validate(dup, teacher, set())
    assert not ok and 'evaluation question' in why, why


def test_clause_that_never_changes_the_answer_is_rejected():
    teacher = SimpleNamespace(solve=lambda task: {'correct': True})
    vacuous = spec()
    vacuous.update(question='Which categories have revenue above {amount}? Return the category and the revenue.',
                   params={'amount': [1, 2]}, skills=['having', 'group_by'], target_skill='having',
                   sql='SELECT p.category, SUM(oi.quantity * oi.unit_price) FROM products p JOIN order_items oi '
                       'ON oi.product_id = p.id GROUP BY p.category HAVING SUM(oi.quantity * oi.unit_price) > {amount}')
    ok, why = evolve.validate(vacuous, teacher, set())
    assert not ok and 'removing HAVING' in why, why


def test_builtin_references_have_no_vacuous_clause():
    import random
    from contextlib import ExitStack, closing
    import env
    with ExitStack() as stack:
        cons = [stack.enter_context(closing(c)) for c in env.verifier_connections()]
        for name in env.BUILTIN:
            pairs = sorted({env.TEMPLATES[name][1](random.Random(s)) for s in range(200)})
            effective = set()
            for _, sql in pairs:
                base = [evolve.signature(env.execute(c, sql)[0]) for c in cons]
                for i, variant in enumerate(evolve.ablations(sql).values()):
                    rows = [env.execute(c, variant)[0] for c in cons]
                    if any(r is None or evolve.signature(r) != b for r, b in zip(rows, base)):
                        effective.add(i)
            assert effective == set(range(len(evolve.ablations(pairs[0][1])))), name


def test_revalidation_keeps_valid_templates_with_long_names(tmp_path, monkeypatch):
    import sqlite3
    db = tmp_path / 'templates.db'
    monkeypatch.setattr(evolve, 'DB', db)
    con = evolve.connect()
    good = spec()
    con.execute("INSERT INTO templates (name, target_skill, skills, tier, question, sql, params, status, reason, model, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)", ('a' * 41, 'filter', '["filter", "aggregate"]', 1, good['question'],
                                                   good['sql'], '{"city": ["London", "Paris", "Tokyo", "Seattle"]}',
                                                   'accepted', 'ok', 'm', 0))
    con.commit()
    evolve.revalidate()
    assert sqlite3.connect(db).execute("SELECT status FROM templates").fetchone()[0] == 'accepted'


def test_part_of_a_heldout_answer_is_rejected_but_unrelated_id_lists_are_not():
    teacher = SimpleNamespace(solve=lambda task: {'correct': True})
    part = spec()
    part.update(question="How many orders with status '{status}' were placed in {year}?",
                params={'status': ['shipped', 'pending'], 'year': [2023, 2024]},
                sql="SELECT COUNT(*) FROM orders WHERE status = '{status}' AND order_date LIKE '{year}-%'")
    ok, why = evolve.validate(part, teacher, set())
    assert not ok and 'overlap the held-out question' in why, why
    ids = spec()
    ids.update(question="List the ids of shipped orders placed in {year}.", params={'year': [2023, 2024]},
               sql="SELECT id FROM orders WHERE status = 'shipped' AND order_date LIKE '{year}-%'")
    ok, why = evolve.validate(ids, teacher, set())
    assert ok, why


def test_proposers_rotate_and_one_failure_keeps_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve, 'DB', tmp_path / 'templates.db')
    calls = []

    def propose(client, model, skill, n, hard=False, *optional):
        calls.append((model, skill, hard))
        if model == 'broken':
            raise RuntimeError('gateway error')
        return [], 0.0

    monkeypatch.setattr(evolve, 'propose', propose)
    monkeypatch.setattr(evolve, 'make_client', lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(evolve, 'Teacher', lambda model, workers: SimpleNamespace(spent=0.0, close=lambda: None, model=model))
    accepted, stats = evolve.evolve(['filter', 'join', 'having'], 2, 'a,broken', solver='s', hard=True, workers=1)
    assert sorted(calls) == sorted([('a', 'filter', True), ('broken', 'join', True), ('a', 'having', True)])
    assert accepted == [] and stats['proposed'] == 0


def test_proposal_parsing_ignores_fences_and_trailing_text(monkeypatch):
    reply = 'Here you go:\n```json\n{"templates": [{"name": "x_y"}]}\n```\nNotes: {not json}'
    monkeypatch.setattr(evolve, 'complete', lambda client, model, prompt: (reply, 0.01))
    prompts = []
    monkeypatch.setattr(evolve, 'value_lists', lambda: '')
    specs, cost = evolve.propose(None, 'm', 'filter', 1)
    assert specs == [{'name': 'x_y', 'target_skill': 'filter', 'model': 'm'}] and cost == 0.01


def test_stepping_stone_prompt_names_the_failing_question(monkeypatch):
    seen = []
    monkeypatch.setattr(evolve, 'complete', lambda client, model, prompt: seen.append(prompt) or ('{"templates": []}', 0.0))
    monkeypatch.setattr(evolve, 'value_lists', lambda: '')
    evolve.propose(None, 'm', 'having', 2, failing=('hard_one', 'Which X?', 'SELECT x FROM t'))
    assert 'It keeps failing this question type' in seen[0] and 'SELECT x FROM t' in seen[0]


def test_reviewer_rejection_blocks_a_valid_template(monkeypatch):
    teacher = SimpleNamespace(solve=lambda task: {'correct': True}, spent=0.0)
    monkeypatch.setattr(evolve, 'complete', lambda client, model, prompt, max_tokens=4096:
                        ('{"verdict": "reject", "reason": "ambiguous city"}', 0.002))
    ok, why = evolve.validate(spec(), teacher, set(), reviewer=(None, 'rev'))
    assert not ok and 'reviewer rev rejected: ambiguous city' in why and teacher.spent == 0.002
    monkeypatch.setattr(evolve, 'complete', lambda client, model, prompt, max_tokens=4096:
                        ('{"verdict": "approve", "reason": "fine"}', 0.0))
    ok, why = evolve.validate(spec(), teacher, set(), reviewer=(None, 'rev'))
    assert ok and 'rev approved' in why


def test_uncovered_combinations_skip_existing_pairs():
    import env
    combos = evolve.uncovered_combinations(1000)
    have = [set(env.SKILLS[t]) for t in env.TEMPLATES if t not in env.HELDOUT_TEMPLATES]
    assert combos and not any(set(c) <= h for c in combos for h in have)


def test_failed_review_rejects_without_crashing(monkeypatch):
    teacher = SimpleNamespace(solve=lambda task: {'correct': True}, spent=0.0)
    def broken(*a, **k):
        raise ValueError('reply hit the token limit')
    monkeypatch.setattr(evolve, 'complete', broken)
    ok, why = evolve.validate(spec(), teacher, set(), reviewer=(None, 'rev'))
    assert not ok and why.startswith('reviewer rev rejected: review failed')


def test_reused_names_get_a_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(evolve, 'DB', tmp_path / 'templates.db')
    specs = evolve.unique_names([{'name': 'count_city'}, {'name': 'new_one'}, {'name': 'new_one'}])
    assert [s['name'] for s in specs] == ['count_city_2', 'new_one', 'new_one_2']


def test_seed_examples_are_shown_to_the_proposer(monkeypatch):
    seen = []
    monkeypatch.setattr(evolve, 'complete', lambda client, model, prompt: seen.append(prompt) or ('{"templates": []}', 0.0))
    monkeypatch.setattr(evolve, 'value_lists', lambda: '')
    seeds = [{'question': 'How many students does each advisor have?', 'query': 'SELECT advisor, count(*) FROM Student GROUP BY advisor'}]
    evolve.propose(None, 'm', 'group_by', 2, seeds=seeds)
    assert 'not their tables' in seen[0] and 'each advisor' in seen[0]
