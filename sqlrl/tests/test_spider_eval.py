import json
import sqlite3
from contextlib import ExitStack, closing

import env
import rollout as engine
import spider_eval


def make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    con.executemany("INSERT INTO singer VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()


def spider_data(tmp_path, monkeypatch):
    """One schema: the original database and a variant where age 30 is shared."""
    folder = tmp_path / "database" / "concert"
    folder.mkdir(parents=True)
    make_db(folder / "concert.sqlite", [(1, "Ava", 30), (2, "Ben", 40)])
    make_db(folder / "concertv1.sqlite", [(1, "Ava", 30), (2, "Ben", 30), (3, "Cy", 50)])
    (tmp_path / "dev.json").write_text(json.dumps([
        {"db_id": "concert", "question": "Names of singers aged 30?", "query": "SELECT name FROM singer WHERE age = 30"},
        {"db_id": "concert", "question": "All singers?", "query": "SELECT * FROM singer CROSS JOIN singer AS b CROSS JOIN singer AS c "
                                                                  "CROSS JOIN singer AS d CROSS JOIN singer AS e CROSS JOIN singer AS f "
                                                                  "CROSS JOIN singer AS g CROSS JOIN singer AS h CROSS JOIN singer AS i"},
    ]))
    monkeypatch.setattr(spider_eval, "DATA", tmp_path)


def test_testsuite_match_allows_column_permutation_but_strict_does_not():
    gold = [("Ava", 30), ("Ben", 40)]
    swapped = [(30, "Ava"), (40, "Ben")]
    assert spider_eval.testsuite_match(swapped, gold, order_matters=False)
    assert not env.results_match(swapped, gold)


def test_testsuite_match_checks_order_only_with_order_by():
    gold = [("Ava",), ("Ben",)]
    assert spider_eval.testsuite_match(list(reversed(gold)), gold, order_matters=False)
    assert not spider_eval.testsuite_match(list(reversed(gold)), gold, order_matters=True)


def test_testsuite_match_keeps_duplicates_and_shape():
    assert not spider_eval.testsuite_match([("Ava",)], [("Ava",), ("Ava",)], order_matters=False)
    assert not spider_eval.testsuite_match([("Ava", 1)], [("Ava",)], order_matters=False)
    assert spider_eval.testsuite_match([], [], order_matters=False)
    assert not spider_eval.testsuite_match(None, [], order_matters=False)
    # A permutation must keep rows together, not just column values.
    assert not spider_eval.testsuite_match([("a", 2), ("b", 1)], [(1, "a"), (2, "b")], order_matters=False)


def test_shop_prompt_is_unchanged_and_other_schemas_replace_it():
    shop = env.system_prompt()
    assert shop.startswith("You are a SQL agent with access to a SQLite database of an online shop. ")
    assert shop.endswith("Database schema:\n" + env.SCHEMA)
    other = env.system_prompt(schema="CREATE TABLE singer (id INTEGER);")
    assert "online shop" not in other and env.SCHEMA not in other
    assert other.startswith("You are a SQL agent with access to a SQLite database. ")
    assert other.endswith("Database schema:\nCREATE TABLE singer (id INTEGER);")


def test_references_that_fail_any_database_are_excluded(tmp_path, monkeypatch):
    spider_data(tmp_path, monkeypatch)
    questions = spider_eval.load_questions()
    expected, excluded = spider_eval.check_references(questions)
    assert list(expected) == [0] and len(expected[0]) == 2
    assert [x["index"] for x in excluded] == [1]


def test_wrong_query_that_passes_the_original_fails_on_a_variant(tmp_path, monkeypatch):
    spider_data(tmp_path, monkeypatch)
    primary, alts = spider_eval.databases("concert")
    assert primary.name == "concert.sqlite" and [p.name for p in alts] == ["concertv1.sqlite"]
    gold = "SELECT name FROM singer WHERE age = 30"
    expected, _ = spider_eval.check_references(spider_eval.load_questions())
    with ExitStack() as stack:
        cons = [stack.enter_context(closing(c)) for c in env.verifier_connections(primary, alts)]
        assert spider_eval.grade_testsuite(cons, "SELECT name FROM singer WHERE age < 35", gold, expected[0])
        assert not spider_eval.grade_testsuite(cons, "SELECT name FROM singer WHERE id = 1", gold, expected[0])
        assert not env.grade(cons, "SELECT name FROM singer WHERE id = 1", gold)[0]


def test_rollout_uses_task_schema_and_variant_databases(tmp_path, monkeypatch):
    spider_data(tmp_path, monkeypatch)
    primary, alts = spider_eval.databases("concert")
    schema = spider_eval.schema_text(primary)
    assert schema.startswith("CREATE TABLE singer")
    prompts = []

    class Tokenizer:
        eos_token_ids = {0}

        def apply_chat_template(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return "prompt"

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, tokens):
            return bytes(tokens).decode()

    tok = Tokenizer()
    monkeypatch.setattr(engine, "_shared_prefix_cache", lambda *a: ([], 1))
    # Correct on the original database only.
    text = '<function name="submit_sql"><param name="query">SELECT name FROM singer WHERE id = 1</param></function>'
    monkeypatch.setattr(engine, "_generate", lambda *a, **k: ([(tok.encode(text) + [0], "stop", [], [0.0])], 1))
    task = env.Task("spider/concert", 0, "Names of singers aged 30?", "SELECT name FROM singer WHERE age = 30",
                    primary, "spider_dev", schema, alts)
    eps, _ = engine.rollout(None, tok, [task])
    assert prompts[0].endswith("Database schema:\n" + schema)
    assert eps[0].outcome == "wrong" and eps[0].verifier_reason == "wrong on alternate database 1"
