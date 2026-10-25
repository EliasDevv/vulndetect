# Reproducibility inventory of the base study

Every parameter relevant to a reported result of the base study, with the state of
its source. Section 4.6 of the thesis summarises this table.

*State* refers to paper, released artefacts and the authors' answers together:
**specified** if the paper states it; **recovered** if the paper is silent but the release
or an answer determines it; **open** if no source determines it, so that the replication
has to choose; **unavailable** if the original step consumed material that is not public;
**contradictory** if the sources determine it differently. Relevance is stated in words;
measured effects are reported in Chapter 7.

Of the six rows marked *contradictory*, Section 4.6 names three (label rule,
fine-tuning instruction, hardware and serving); the line breaks of Figure 1 (row 6)
are handled in Section 4.3.

| # | Area | Gap | State | Relevance |
|---|---|---|---|---|
| 1 | Data | Proprietary portion of training and test data | unavailable | bounds every absolute number |
| 2 | Data | Curation logic and split procedure | unavailable | instance sets not comparable |
| 3 | Data | Expert review of the labels | unavailable | label noise cannot be assessed |
| 4 | Data | Validation split | open | none for the reported numbers |
| 5 | Prompt | Which prompt produced the baseline table | contradictory | every baseline metric |
| 6 | Prompt | Line breaks of Figure 1 | contradictory | verdicts, via formatting sensitivity |
| 7 | Prompt | Content of the CVE field | contradictory (resolved by the answers) | a label-proximal input |
| 8 | Prompt | Fine-tuning instruction differs from Figure 1 | contradictory | what the SFT adapter is trained on |
| 9 | Inference | Model variant | recovered | answer style throughout |
| 10 | Inference | Checkpoint revision | open | metadata only |
| 11 | Inference | Decoding parameters | recovered in part | every metric |
| 12 | Inference | Serving stack and precision | contradictory | every metric |
| 13 | Labels | Rule from answer to label | contradictory | every metric |
| 14 | Labels | Exclusion of unreadable answers | recovered | denominators of every metric |
| 15 | RAG | Embedder, re-ranker, final set, distance, query, context position | recovered | retrieved content and its use |
| 16 | RAG | Document collection and its filter | unavailable | content of the knowledge base |
| 17 | RAG | Category pages in the collection | unavailable | two of the five target categories |
| 18 | SFT | Library, batch, sequence length, target modules, bits, serialisation, loss | recovered | what is trained, and on how much data |
| 19 | SFT | Evaluation state of the adapter | recovered | plausibly the SFT numbers |
| 20 | Dual-Agent | Validator prompt | recovered | the approach as a whole |
| 21 | Dual-Agent | Decision rule and number of rounds | open | every Dual-Agent metric |
| 22 | Evaluation | Positive class, overall average | recovered | every metric |
| 23 | Evaluation | Tail of the t-test | open | significance statements |
| 24 | Evaluation | Seeds and repetitions | recovered | run-to-run variance unknown |
