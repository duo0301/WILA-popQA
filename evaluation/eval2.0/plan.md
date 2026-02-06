The script currently does:

Fetch country Q-numbers via P27 for each entity
Walk P1366 to terminal successors, fetch P1549 demonyms + aliases + labels, cache everything
Normalize raw LLM output (strip noise, handle entity leakage, slash alternates)
Match candidates against expanded GT using the priority scale (exact → alias → substring → demonym)
Flag systematic_ar and systematic_zh, score everything else

What's missing:
After Stage 1 matching, collect all rows that scored 0.0 and have no systematic flag. Send those to a frontier model (not in the evaluation set) as judge. The judge receives the GT set, the raw output, and the original question. It classifies each row as either correct (with a reason: correct_demonym, correct_common_name) or error (with a type from the taxonomy). Correct rows get score 0.7. Errors stay at 0.0 and get the error type written to a column.