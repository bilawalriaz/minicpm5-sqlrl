Spider dev (spider-dev-testsuite-v1), 1,032 questions, 4 attempts each, temperature 1, 5 turns, 384 tokens per turn.
Runs: `runs/spider_1b_released`, `runs/spider_1b_pipeline`, `runs/spider_2b_released`, `runs/spider_2b_pipeline`.

| Model | Test-suite pass@1 | Strict pass@1 | Test-suite pass@4 |
| --- | --- | --- | --- |
| Released 1B | 6.6% (6%-8%) | 6.5% | 17.2% |
| Fine-tuned 1B | 29.2% (27%-32%) | 28.3% | 44.4% |
| Released 2B | 41.6% (39%-44%) | 38.7% | 58.8% |
| Fine-tuned 2B | 54.1% (52%-57%) | 50.5% | 68.8% |

| Database | Questions | Released 1B | Fine-tuned 1B | Released 2B | Fine-tuned 2B |
| --- | --- | --- | --- | --- | --- |
| world_1 | 120 | 2% | 13% | 24% | 30% |
| car_1 | 92 | 1% | 20% | 28% | 32% |
| cre_Doc_Template_Mgt | 84 | 10% | 32% | 53% | 65% |
| dog_kennels | 82 | 2% | 22% | 46% | 60% |
| flight_2 | 80 | 10% | 26% | 27% | 44% |
| student_transcripts_tracking | 78 | 2% | 24% | 33% | 40% |
| tvshow | 62 | 6% | 39% | 50% | 62% |
| wta_1 | 60 | 4% | 31% | 40% | 55% |
| network_1 | 56 | 13% | 45% | 43% | 59% |
| concert_singer | 45 | 4% | 34% | 43% | 62% |
| pets_1 | 42 | 2% | 17% | 58% | 74% |
| orchestra | 40 | 6% | 47% | 48% | 72% |
| poker_player | 40 | 14% | 38% | 61% | 78% |
| employee_hire_evaluation | 38 | 9% | 51% | 55% | 75% |
| course_teach | 30 | 11% | 28% | 46% | 57% |
| singer | 30 | 21% | 38% | 62% | 68% |
| museum_visit | 18 | 19% | 44% | 58% | 76% |
| battle_death | 16 | 8% | 20% | 34% | 48% |
| voter_1 | 15 | 13% | 53% | 68% | 70% |
| real_estate_properties | 4 | 6% | 25% | 25% | 38% |
