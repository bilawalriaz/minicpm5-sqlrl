# Testing the fine-tuned MiniCPM5-1B SQL agent in LM Studio

Models (already in LM Studio under `bilawalriaz/`):

- `MiniCPM5-1B-SQL-GGUF/MiniCPM5-1B-SQL-Q8_0.gguf`: fine-tuned, 8-bit, 1.15 GB
- `MiniCPM5-1B-SQL-GGUF/MiniCPM5-1B-SQL-Q4_K_M.gguf`: fine-tuned, 4-bit, 0.69 GB
- `MiniCPM5-1B-released-GGUF/MiniCPM5-1B-released-Q8_0.gguf`: OpenBMB's released model, for comparison

Both builds default to no-think mode, which is how the model was trained and evaluated.

## Settings

| Setting | Value |
| --- | --- |
| Temperature | 0 (repeatable); 0.7-1.0 to see the spread of answers |
| Top P | 1.0 |
| Min P | 0 |
| Repeat penalty | 1.0 (off; a penalty breaks SQL, which repeats table names) |
| Max response tokens | 384 |
| Context length | 4096 |
| System prompt | the whole of `lmstudio-system-prompt.txt` |

Don't enable LM Studio's tool use. The model writes its own calls in the MiniCPM format:
`<function name="run_sql"><param name="query">SELECT ...</param></function>`. LM Studio may hide the
`<function` and `<param` tags because they are special tokens; the SQL still shows.

## Running a question

1. Paste the system prompt, then ask a question from `lmstudio-prompts.txt`.
2. If the reply is `run_sql`, run the query on the evaluation database and paste the result back as
   the next user message, exactly like this:

       <tool_response>
       40
       </tool_response>

   Query it with `sqlite3 sqlrl/data/sql-v2.2/eval.db "SELECT ..."` from the repository.
3. A `submit_sql` reply is the final answer. Run that query and compare it with the expected answer.

Most of the time the fine-tuned model checks once with `run_sql` and then submits the same query, so
the first query already tells you whether it got the question right.

The percentages in `lmstudio-prompts.txt` are the fine-tuned model's success rates over 16 sampled attempts
in the evaluation. Questions 1-7 are from the test set and 8-9 from the check set; the model never
trained on those question types. Question 10 is a trained type with values it never saw.
