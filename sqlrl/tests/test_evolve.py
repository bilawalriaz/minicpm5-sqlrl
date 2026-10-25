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
