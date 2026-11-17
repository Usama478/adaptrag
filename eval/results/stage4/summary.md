# Stage 4 mini evaluation (dev documents)

Embedding model BAAI/bge-small-en-v1.5, parser 0.2.0, top 10, exact search. Intervals: cluster bootstrap over documents, 10000 resamples, seed 20261116.

## Results (machine-verified questions)

| row | R@1 | R@5 | R@10 | MRR | 95% R@5 | 95% MRR | N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_size | 0.511 | 0.867 | 0.944 | 0.659 | [0.778, 0.956] | [0.547, 0.760] | 90 |
| recursive | 0.567 | 0.844 | 0.900 | 0.691 | [0.756, 0.933] | [0.592, 0.790] | 90 |
| section_aware | 0.500 | 0.844 | 0.911 | 0.653 | [0.744, 0.933] | [0.590, 0.719] | 90 |
| router (rule v1) | 0.489 | 0.844 | 0.911 | 0.647 | [0.744, 0.933] | [0.588, 0.708] | 90 |
| best single (fixed_size) | 0.511 | 0.867 | 0.944 | 0.659 | [0.778, 0.956] | [0.547, 0.760] | 90 |
| oracle (upper bound) | 0.600 | 0.900 | 0.933 | 0.721 | [0.822, 0.978] | [0.625, 0.814] | 90 |

Best single strategy and oracle are chosen on these same dev numbers.

## Table token budget variants

| row | R@1 | R@5 | R@10 | MRR | 95% R@5 | 95% MRR | N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| recursive+tb512 | 0.556 | 0.844 | 0.889 | 0.682 | [0.756, 0.933] | [0.586, 0.779] | 90 |
| section_aware+tb512 | 0.500 | 0.844 | 0.922 | 0.652 | [0.744, 0.933] | [0.589, 0.720] | 90 |

## Owner-verified paper_01 set (10 questions, one document)

| row | R@1 | R@5 | R@10 | MRR | 95% R@5 | 95% MRR | N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| owner paper_01: fixed_size | 0.700 | 0.900 | 1.000 | 0.814 | n/a (1 document) | n/a (1 document) | 10 |
| owner paper_01: recursive | 0.700 | 1.000 | 1.000 | 0.833 | n/a (1 document) | n/a (1 document) | 10 |
| owner paper_01: section_aware (routed) | 0.900 | 0.900 | 1.000 | 0.912 | n/a (1 document) | n/a (1 document) | 10 |

## Per question type

| type | row | N | hits@5 | MRR |
| --- | --- | --- | --- | --- |
| factual | fixed_size | 60 | 51/60 | 0.609 |
| factual | recursive | 60 | 51/60 | 0.700 |
| factual | section_aware | 60 | 50/60 | 0.668 |
| factual | router | 60 | 50/60 | 0.668 |
| table | fixed_size | 6 | 4/6 | 0.560 |
| table | recursive | 6 | 2/6 | 0.333 |
| table | section_aware | 6 | 3/6 | 0.278 |
| table | router | 6 | 3/6 | 0.278 |
| multi_section | fixed_size | 24 | 23/24 | 0.809 |
| multi_section | recursive | 24 | 23/24 | 0.758 |
| multi_section | section_aware | 24 | 23/24 | 0.709 |
| multi_section | router | 24 | 23/24 | 0.688 |

## Per document: hits@5 (MRR), routed and best strategy

| document | category | N | fixed_size | recursive | section_aware | routed | best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper_01 | sectioned | 10 | 9/10 (0.766) | 10/10 (0.950) | 9/10 (0.812) | section_aware | recursive |
| paper_02 | sectioned | 10 | 7/10 (0.462) | 6/10 (0.460) | 7/10 (0.534) | section_aware | section_aware |
| paper_03 | sectioned | 10 | 7/10 (0.470) | 8/10 (0.524) | 6/10 (0.591) | section_aware | recursive |
| paper_04 | sectioned | 10 | 8/10 (0.642) | 7/10 (0.617) | 8/10 (0.542) | section_aware | fixed_size |
| paper_05 | short | 10 | 10/10 (0.803) | 10/10 (0.833) | 10/10 (0.692) | section_aware | recursive |
| paper_06 | sectioned | 10 | 10/10 (0.858) | 9/10 (0.817) | 10/10 (0.587) | section_aware | fixed_size |
| paper_07 | short | 10 | 10/10 (0.758) | 8/10 (0.683) | 9/10 (0.717) | section_aware | fixed_size |
| doc_08 | poorly_structured | 10 | 10/10 (0.783) | 10/10 (0.750) | 10/10 (0.800) | recursive | section_aware |
| paper_09 | table_heavy | 10 | 7/10 (0.392) | 8/10 (0.587) | 7/10 (0.600) | section_aware | recursive |

## Router rule v1

Rule fixed before the run: propose a rule v2 only if the routed strategy loses on at least 3 of the documents (another strategy has more hits@5) AND one profile threshold separates those documents.
Routed strategy loses on 4 documents: paper_01, paper_03, paper_07, paper_09.
Outcome: **KEEP v1** (the router is not changed in this stage).

## Paired outcomes per question (top 5)

| pair | only first | only second | both | neither |
| --- | --- | --- | --- | --- |
| fixed_size vs recursive | 7 | 5 | 71 | 7 |
| fixed_size vs section_aware | 6 | 4 | 72 | 8 |
| recursive vs section_aware | 4 | 4 | 72 | 10 |

## Ties

TIE: recursive (0.844) and section_aware (0.844) are within 0.01 on Recall@5. Found only by recursive: paper_01_r0_d12, paper_03_r0_d02, paper_03_r1_d18, paper_09_r0_d10. Found only by section_aware: paper_02_r0_d03, paper_04_r0_d10, paper_06_r1_d19, paper_07_r0_d14.

## Evidence split across chunk boundaries

split = no chunk of the set holds an evidence entry whole, but one holds at least half of it. missing = no chunk of the set is relevant at all.

| variant | questions split | questions missing |
| --- | --- | --- |
| fixed_size | 0 | 0 |
| recursive | 0 | 0 |
| section_aware | 0 | 0 |
| recursive+tb512 | 0 | 0 |
| section_aware+tb512 | 0 | 0 |

## Glued-word candidates (general detector)

A candidate is a run of letters that is not in the embedding model's vocabulary and splits into two vocabulary words (two words), or has a lowercase letter followed by an uppercase one (case join). Candidates include real compounds and product names.

| document | candidates | distinct | two words | case join | most frequent |
| --- | --- | --- | --- | --- | --- |
| paper_01 | 331 | 137 | 325 | 6 | arXiv, Transformer, decoder, preprint, recurrent, positional, CoRR, GPUs |
| paper_02 | 499 | 141 | 453 | 46 | ResNet, ImageNet, mAP, shortcuts, dataset, shortcut, ResNets, bottleneck |
| paper_03 | 199 | 97 | 195 | 4 | Hinton, multilayer, ICML, Goodfellow, backpropagation, CIFAR, MNIST, Parzen |
| paper_04 | 542 | 209 | 496 | 46 | OpenAI, bidirectional, BERTBASE, SQuAD, Transformer, BERTLARGE, ELMo, MNLI |
| paper_05 | 94 | 40 | 85 | 9 | blockchain, traceability, Blockchain, POMS, ZoKrates, concealing, decrypt, encrypt |
| paper_06 | 160 | 89 | 149 | 11 | Dropout, ImageNet, LeCun, backpropagation, pooling, CoRR, Hinton, ReLU |
| paper_07 | 186 | 109 | 142 | 44 | DistilBERT, ArXiv, SQuAD, ELMo, benchmark, IMDb, Transformer, distilled |
| doc_08 | 71 | 52 | 71 | 0 | repurchases, CEOs, GAAP, Amex, BNSF, unmatched, Tailwind, yearend |
| paper_09 | 1632 | 531 | 1162 | 470 | SGPT, arXiv, datasets, msmarco, MassiveScenarioClassification, MassiveIntentClassification, huggingface, MiniLM |

Questions whose evidence contains a candidate: 51 of 90 machine-verified, 3 of 10 owner-verified.

## Table token budget (512 tokens)

| strategy | table chunks over 512, no budget | with budget | table hits@5, no budget | with budget |
| --- | --- | --- | --- | --- |
| recursive | 25/125 | 0/151 | 2/6 | 2/6 |
| section_aware | 30/126 | 0/156 | 3/6 | 3/6 |

Rule fixed before the run: ADOPT if table chunks over 512 tokens drop to 0 AND hits@5 on table questions drop by no more than 1 question, for both strategies; otherwise KEEP.
Outcome: **ADOPT** (the default changes only in a follow-up).

## Very short documents

| document | words | pages |
| --- | --- | --- |
| paper_01 | 5990 | 15 |
| paper_02 | 9507 | 12 |
| paper_03 | 4502 | 9 |
| paper_04 | 9719 | 16 |
| paper_05 | 3177 | 4 |
| paper_06 | 7082 | 11 |
| paper_07 | 2733 | 5 |
| doc_08 | 3592 | 10 |
| paper_09 | 17216 | 24 |

Rule fixed before the run: no "very short" rule unless at least 2 dev documents are under 2000 words. Under the limit: none. Outcome: **NO RULE**.

## Top-1 scores under the router, and the not-found threshold

| top-1 score | answerable questions | off-topic questions |
| --- | --- | --- |
| 0.30-0.35 | 0 | 2 |
| 0.35-0.40 | 0 | 20 |
| 0.40-0.45 | 0 | 80 |
| 0.45-0.50 | 0 | 58 |
| 0.50-0.55 | 1 | 16 |
| 0.55-0.60 | 1 | 3 |
| 0.60-0.65 | 10 | 1 |
| 0.65-0.70 | 19 | 0 |
| 0.70-0.75 | 26 | 0 |
| 0.75-0.80 | 17 | 0 |
| 0.80-0.85 | 12 | 0 |
| 0.85-0.90 | 3 | 0 |
| 0.90-0.95 | 1 | 0 |

answerable: n 90, min 0.548, median 0.728, max 0.901.
off-topic: n 180, min 0.320, median 0.443, max 0.607.
Current retrieval_min_score 0.55: 1 answerable questions fall below it, 4 off-topic questions pass it.
Proposal: 0.567 puts 268 of 270 scores on the right side (answerable at or above, off-topic below). The setting is NOT changed here: the owner decides. Off-topic questions are the easy case; in-domain questions without an answer score much higher (Stage 2 notes).

## Section-aware: boundaries or prefix?

Stage 4 cannot separate the two. Section-aware chunks differ from recursive chunks in where they are cut AND in the heading prefix that is embedded with them. The relevance rule ignores the prefix, but retrieval does not. The Stage 5 switch (the same boundaries with and without the prefix) will separate them.

## Chunk boundaries: paper_01 (routed: section_aware), first 15 chunks of each strategy

| # | fixed_size | recursive | section_aware |
| --- | --- | --- | --- |
| 0 | p1, 200w: Provided proper attribution is provided, Google hereby grant | p1, 72w: Provided proper attribution is provided, Google hereby grant | p1, 26w: Provided proper attribution is provided, Google hereby grant |
| 1 | p1, 200w: 2014 Englishto-German translation task, improving over the e | p1, 190w: Llion Jones_∗_ Aidan N. Gomez_∗†_ Łukasz Kaiser_∗_ Google Re | p1, 45w: Attention Is All You Need Ashish Vaswani_∗_ Noam Shazeer_∗_  |
| 2 | p1-2, 200w: tensor2tensor. Llion also experimented with novel model vari | p1-2, 155w: > ∗ Equal contribution. Listing order is random. Jakob propo | p1, 176w: Attention Is All You Need > Abstract The dominant sequence t |
| 3 | p2, 200w: generate a sequence of hidden states ht , as a function of t | p1-2, 198w: > † Work performed while at Google Brain. > ‡ Work performed | p1, 160w: Attention Is All You Need > Abstract > ∗ Equal contribution. |
| 4 | p2, 200w: between input and output. The Transformer allows for signifi | p2, 115w: Attention mechanisms have become an integral part of compell | p2, 175w: Attention Is All You Need > 1 Introduction Recurrent neural  |
| 5 | p2, 200w: 3.2. Self-attention, sometimes called intra-attention is an  | p2, 175w: 2 Background The goal of reducing sequential computation als | p2, 121w: Attention Is All You Need > 1 Introduction Attention mechani |
| 6 | p2-3, 200w: symbol representations ( x 1 , ..., xn ) to a sequence of co | p2-3, 184w: End-to-end memory networks are based on a recurrent attentio | p2, 181w: Attention Is All You Need > 2 Background The goal of reducin |
| 7 | p3, 200w: followed by layer normalization [1]. That is, the output of  | p3, 147w: Figure 1: The Transformer - model architecture. The Transfor | p2, 93w: Attention Is All You Need > 2 Background End-to-end memory n |
| 8 | p3-4, 200w: at positions less than i . 3.2 Attention An attention functi | p3-4, 199w: Decoder: The decoder is also composed of a stack of N = 6 id | p2-3, 136w: Attention Is All You Need > 3 Model Architecture Most compet |
| 9 | p4, 200w: queries simultaneously, packed together into a matrix Q . Th | p4, 137w: of the values, where the weight assigned to each value is co | p3, 119w: Attention Is All You Need > 3 Model Architecture > 3.1 Encod |
| 10 | p4-5, 200w: the softmax function into regions where it has extremely sma | p4, 158w: Attention( Q, K, V ) = softmax( QK T ) V (1) √ dk The two mo | p3, 126w: Attention Is All You Need > 3 Model Architecture > 3.1 Encod |
| 11 | p5, 200w: to information from different representation subspaces at di | p4-5, 179w: 3.2.2 Multi-Head Attention Instead of performing a single at | p3, 49w: Attention Is All You Need > 3 Model Architecture > 3.2 Atten |
| 12 | p5, 200w: every position in the decoder to attend over all positions i | p5, 172w: MultiHead( Q, K, V ) = Concat(head1 , ..., headh) W O where  | p4, 55w: Attention Is All You Need > 3 Model Architecture > Scaled Do |
| 13 | p5-6, 200w: attention sub-layers, each of the layers in our encoder and  | p5, 186w: - The encoder contains self-attention layers. In a self-atte | p4, 198w: Attention Is All You Need > 3 Model Architecture > 3.2.1 Sca |
| 14 | p5-6, 200w: weight matrix between the two embedding layers and the pre-s | p5-6, 192w: FFN( x ) = max(0 , xW 1 + b 1) W 2 + b 2 (2) While the linea | p4, 81w: Attention Is All You Need > 3 Model Architecture > 3.2.1 Sca |

Chunks in total: fixed_size 43, recursive 45, section_aware 56.

## Chunk boundaries: doc_08 (routed: recursive), first 15 chunks of each strategy

| # | fixed_size | recursive | section_aware |
| --- | --- | --- | --- |
| 0 | p1, 200w: Berkshire’s Performance vs. the S&P 500 \| \| Annual Percen \|  | p1, 6w: Berkshire’s Performance vs. the S&P 500 | p1, 198w: Berkshire’s Performance vs. the S&P 500 \| \| Annual Percen \|  |
| 1 | p1, 200w: . . . . . . . . . . . . . . \| . . . . . . . . . . . . . . .  | p1, 192w: \| \| Annual Percen \| tage Change \| \| --- \| --- \| --- \| \| \| in | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1967 . . . . . . . |
| 2 | p1, 200w: . . . . . . . . . . . . \| . . . . . . . . . . . . . . . . .  | p1, 158w: \| 1967 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1969 . . . . . . . |
| 3 | p1, 200w: . . . . . . . . . . \| . . . . . . . . . . . . . . . . . . .  | p1, 158w: \| 1969 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1971 . . . . . . . |
| 4 | p1, 200w: . . . . . . . . \| . . . . . . . . . . . . . . . . . . . .<br | p1, 158w: \| 1971 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1973 . . . . . . . |
| 5 | p1, 200w: . . . . . . \| . . . . . . . . . . . . . . . . . . . .<br>(48 | p1, 158w: \| 1973 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 157w: Berkshire’s Performance vs. the S&P 500 \| 1975 . . . . . . . |
| 6 | p1, 200w: . . . . . . . . . . . . . . . . . .<br>129.3 \| 37.2<br>23.6  | p1, 151w: \| 1975 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 171w: Berkshire’s Performance vs. the S&P 500 \| 1977 . . . . . . . |
| 7 | p1, 200w: . . . . . . . . . . . . . . . .<br>14.5 \| 6.4 \| \| 1979 \| <br | p1, 165w: \| 1977 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 156w: Berkshire’s Performance vs. the S&P 500 \| . . . . . . . . .  |
| 8 | p1, 200w: . . . . . . . . . . . . .<br>32.8 \| .<br>32.3 \| \| 1981 . . . | p1, 150w: \| . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1981 . . . . . . . |
| 9 | p1, 200w: . . . . . . . . . . .<br>38.4 \| 21.4 \| \| 1983 . . . . . . .  | p1, 158w: \| 1981 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1983 . . . . . . . |
| 10 | p1, 200w: . . . . . . . . .<br>(2.7) \| 6.1 \| \| 1985 . . . . . . . . .  | p1, 158w: \| 1983 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 85w: Berkshire’s Performance vs. the S&P 500 \| 1985 . . . . . . . |
| 11 | p1, 200w: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | p1, 79w: \| 1985 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 157w: Berkshire’s Performance vs. the S&P 500 \| 1986 . . . . . . . |
| 12 | p1, 200w: \| \| 1989 . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 151w: \| 1986 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1988 . . . . . . . |
| 13 | p1, 200w: 1991 . . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 158w: \| 1988 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1990 . . . . . . . |
| 14 | p1, 200w: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | p1, 158w: \| 1990 . . . . . . . . . . . . . . . . . . . . . . . . . . . | p1, 164w: Berkshire’s Performance vs. the S&P 500 \| 1992 . . . . . . . |

Chunks in total: fixed_size 52, recursive 56, section_aware 57.

## Failed questions under the router (not in the top 5)

### paper_01_r0_d12 (factual, paper_01, routed: section_aware)

- Question: Who are the authors of the paper titled 'Building a large annotated corpus of english: The penn treebank'?
- Answer: Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini
- Evidence p12: Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank.
- Router rank of the first relevant chunk: 8
- Found in the top 5 by: fixed_size, recursive
- Retrieved 1: score 0.636, p10: Attention Is All You Need > 6 Results > 6.3 English Constituency Parsing increased the maximum output length to input length + 300. We used a beam size of 21 and α = 0 . 3 for both WSJ only and the se
- Retrieved 2: score 0.626, p9-10: Attention Is All You Need > 6 Results > 6.3 English Constituency Parsing To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This tas
- Retrieved 3: score 0.589, p12: Attention Is All You Need > 7 Conclusion - [27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing , 

### paper_02_r0_d05 (factual, paper_02, routed: section_aware)

- Question: What is the number of layers in ResNet-34?
- Answer: 34 layers
- Evidence p5: 50 50 40 40 34-layer 18-layer 30 30 18-layer plain-18 ResNet-18 plain-34 ResNet-34 34-layer 20 20 0 10 20 30 40 50 0 10 20
- Router rank of the first relevant chunk: 7
- Found in the top 5 by: no strategy
- Retrieved 1: score 0.762, p6-7: Deep Residual Learning for Image Recognition > 4. Experiments > 4.1. ImageNet Classification 50-layer ResNet: We replace each 2-layer block in the > 4Deeper non -bottleneck ResNets ( e.g ., Fig. 5 lef
- Retrieved 2: score 0.747, p8: Deep Residual Learning for Image Recognition > 4. Experiments > 4.2. CIFAR-10 and Analysis Figure 7. Standard deviations (std) of layer responses on CIFAR10. The responses are the outputs of each 3 × 
- Retrieved 3: score 0.745, p5: Deep Residual Learning for Image Recognition > 4. Experiments > 4.1. ImageNet Classification reducing of the training error3 . The reason for such optimization difficulties will be studied in the futu

### paper_02_r0_d09 (factual, paper_02, routed: section_aware)

- Question: What is the size of the 1202-layer network mentioned?
- Answer: 19.4M
- Evidence p8: the 1202-layer network may be unnecessarily large (19.4M) for this small dataset.
- Evidence p7: \| \| ResNet \| 110 \| 1.7M \| **6.43**(6.61_±_0.16) \| \| ResNet \| 1202 \| 19.4M \| 7.93 \|
- Router rank of the first relevant chunk: none in the top 10
- Found in the top 5 by: no strategy
- Retrieved 1: score 0.642, p8: Deep Residual Learning for Image Recognition > 4. Experiments > 4.2. CIFAR-10 and Analysis Exploring Over 1000 layers. We explore an aggressively deep model of over 1000 layers. We set n = 200 that le
- Retrieved 2: score 0.636, p4: Deep Residual Learning for Image Recognition > 3. Deep Residual Learning > 3.3. Network Architectures Figure 3. Example network architectures for ImageNet. Left : the VGG-19 model [41] (19.6 billion F
- Retrieved 3: score 0.625, p10: Deep Residual Learning for Image Recognition > A. Object Detection Baselines Unlike VGG-16 used in [32], our ResNet has no hidden fc layers. We adopt the idea of “Networks on Conv feature maps” (NoC) 

### paper_02_r0_d17 (table, paper_02, routed: section_aware)

- Question: What is the mAP value for the baseline+++ system using ResNet-101?
- Answer: 83.8
- Evidence p11: \| baseline+++ \| ResNet-101 \| COCO+07++12 \| **83.8** \| **92.1 88.4 84.8 75.9 71.4 86.3 87.8 94.2 66.8 89.4 69.2 93.9 91.9 90.9 89.6 67.9 88.2 76.8 90.3 80.0** \|
- Router rank of the first relevant chunk: none in the top 10
- Found in the top 5 by: no strategy
- Retrieved 1: score 0.733, p10: Deep Residual Learning for Image Recognition > A. Object Detection Baselines > MS COCO Table 8 shows the results on the MS COCO validation set. ResNet-101 has a 6% increase of mAP@[.5, .95] over VGG-1
- Retrieved 2: score 0.714, p12: Deep Residual Learning for Image Recognition > C. ImageNet Localization Table 14. Comparisons of localization error (%) on the ImageNet dataset with state-of-the-art methods. ports a center-crop error
- Retrieved 3: score 0.704, p8: Deep Residual Learning for Image Recognition > 4. Experiments > 4.2. CIFAR-10 and Analysis Figure 7. Standard deviations (std) of layer responses on CIFAR10. The responses are the outputs of each 3 × 

### paper_03_r0_d02 (factual, paper_03, routed: section_aware)

- Question: What is the role of the generative model in the adversarial nets framework?
- Answer: to produce fake currency
- Evidence p1: the generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency
- Router rank of the first relevant chunk: 10
- Found in the top 5 by: fixed_size, recursive
- Retrieved 1: score 0.833, p8: Generative Adversarial Nets > 7 Conclusions and future work 5. Efficiency improvements: training could be accelerated greatly by divising better methods for coordinating G and D or determining better 
- Retrieved 2: score 0.820, p6: Generative Adversarial Nets > 5 Experiments \| Model \| MNIST \| TFD \| \| --- \| --- \| --- \| \| DBN [3] \| 138_±_2 \| 1909_±_66 \| \| Stacked CAE [3] \| 121_±_1_._6 \| **2110**_±_**50** \| \| Deep GSN [6] \| 214_±_1
- Retrieved 3: score 0.818, p1: Generative Adversarial Nets > Abstract We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that ca

### paper_03_r0_d08 (factual, paper_03, routed: section_aware)

- Question: What value of k was used in the experiments for the discriminator steps?
- Answer: 1
- Evidence p4: We used k = 1, the least expensive option, in our experiments.
- Router rank of the first relevant chunk: 6
- Found in the top 5 by: fixed_size
- Retrieved 1: score 0.647, p4: Generative Adversarial Nets > 4 Theoretical Results - for number of training iterations do for k steps do - Sample minibatch of m noise samples { z(1) , . . . , z(m) } from noise prior pg ( z ). - Sam
- Retrieved 2: score 0.645, p4: Generative Adversarial Nets > 4 Theoretical Results > 4.1 Global Optimality of pg = p data We first consider the optimal discriminator D for any given generator G . Proposition 1. For G fixed, the opt
- Retrieved 3: score 0.610, p5: Generative Adversarial Nets > 4 Theoretical Results > 4.1 Global Optimality of pg = p data Theorem 1. The global minimum of the virtual training criterion C ( G ) is achieved if and only if pg = pdata

### paper_03_r0_d12 (factual, paper_03, routed: section_aware)

- Question: What type of model is suggested to be reasonable for use despite lacking theoretical guarantees?
- Answer: multilayer perceptrons
- Evidence p5: the excellent performance of multilayer perceptrons in practice suggests that they are a reasonable model to use despite their lack of theoretical guarantees.
- Evidence p1: recovering the training data distribution and D equal to 21everywhere.In the case where_G_and_D_are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov
- Router rank of the first relevant chunk: none in the top 10
- Found in the top 5 by: no strategy
- Retrieved 1: score 0.643, p7-8: Generative Adversarial Nets > 7 Conclusions and future work This framework admits many straightforward extensions: > 1. A conditional generative model p ( x \| c ) can be obtained by adding c as input 
- Retrieved 2: score 0.639, p2: Generative Adversarial Nets > 2 Related work An alternative to directed graphical models with latent variables are undirected graphical models with latent variables, such as restricted Boltzmann machi
- Retrieved 3: score 0.639, p6: Generative Adversarial Nets > 5 Experiments Table 1: Parzen window-based log-likelihood estimates. The reported numbers on MNIST are the mean loglikelihood of samples on test set, with the standard er

### paper_03_r1_d18 (multi_section, paper_03, routed: section_aware)

- Question: What is the main focus of the proposed adversarial nets framework?
- Answer: generative model against an adversary
- Evidence p1: The generative model is pitted against an adversary: a discriminative model
- Evidence p6: this method of estimating the likelihood has somewhat high variance
- Router rank of the first relevant chunk: 7
- Found in the top 5 by: recursive
- Retrieved 1: score 0.777, p8: Generative Adversarial Nets > 7 Conclusions and future work 5. Efficiency improvements: training could be accelerated greatly by divising better methods for coordinating G and D or determining better 
- Retrieved 2: score 0.772, p2-3: Generative Adversarial Nets > 3 Adversarial nets The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons. To learn the generator’s distribut
- Retrieved 3: score 0.769, p1: Generative Adversarial Nets Ian J. Goodfellow, Jean Pouget-Abadie_∗_ , Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair_†_ , Aaron Courville, Yoshua Bengio_‡_ D´epartement d’informatique et de 

### paper_04_r0_d02 (factual, paper_04, routed: section_aware)

- Question: What does BERT use for pre-training that differs from Radford et al. (2018)?
- Answer: masked language models
- Evidence p2: BERT uses masked language models to enable pretrained deep bidirectional representations.
- Router rank of the first relevant chunk: none in the top 10
- Found in the top 5 by: no strategy
- Retrieved 1: score 0.765, p14: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding > A Additional Details for BERT > A.4 Comparison of BERT, ELMo ,and OpenAI GPT Here we studies the differences in recen
- Retrieved 2: score 0.761, p7: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding > 4 Experiments > 4.2 SQuAD v1.1 The results compared to prior leaderboard entries and top published work (Sun et al., 
- Retrieved 3: score 0.759, p5: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding > 3 BERT > 3.1 Pre-training BERT Pre-training data The pre-training procedure largely follows the existing literature o

### paper_04_r0_d09 (factual, paper_04, routed: section_aware)

- Question: What is the F1 score difference between the best performing method and fine-tuning the entire model?
- Answer: 0.3
- Evidence p9: only 0.3 F1 behind fine-tuning the entire model.
- Router rank of the first relevant chunk: none in the top 10
- Found in the top 5 by: no strategy
- Retrieved 1: score 0.652, p13: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding > A Additional Details for BERT > A.3 Fine-tuning Procedure For fine-tuning, most model hyperparameters are the same as
- Retrieved 2: score 0.642, p6: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding > 4 Experiments > 4.1 GLUE Table 1: GLUE Test results, scored by the evaluation server (https://gluebenchmark.com/leade
- Retrieved 3: score 0.628, p6: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding > 4 Experiments > 4.2 SQuAD v1.1 Table 2 shows top leaderboard entries as well as results from top published systems (S

### paper_07_r1_d14 (table, paper_07, routed: section_aware)

- Question: What is the score for BERT-base on the MNLI task?
- Answer: 86.7
- Evidence p3: \| BERT-base \| 79.5 \| 56.3 \| 86.7 \| 88.6 \| 91.8 \| 89.6 \| 69.3 \| 92.7 \| 89.0 \| 53.5 \|
- Router rank of the first relevant chunk: 6
- Found in the top 5 by: fixed_size
- Retrieved 1: score 0.728, p4: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter > 4 Experiments > 4.1 Downstream task benchmark \| Ablation \| **Variation on GLUE macro-score** \| \| --- \| --- \| \| _∅_-_Lcos
- Retrieved 2: score 0.720, p3-4: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter > 4 Experiments > 4.1 Downstream task benchmark Downstream tasks We further study the performances of DistilBERT on severa
- Retrieved 3: score 0.708, p3: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter > 4 Experiments General Language Understanding We assess the language understanding and generalization capabilities of Dis

### paper_09_r0_d04 (factual, paper_09, routed: section_aware)

- Question: What is the highest score recorded in the first passage?
- Answer: 96
- Evidence p4: 93 93 87 90 89 90 96 96 95 95
- Evidence p20: 96 94 92 98 97 95 91 94
- Router rank of the first relevant chunk: none in the top 10
- Found in the top 5 by: no strategy
- Retrieved 1: score 0.548, p18: MTEB: Massive Text Embedding Benchmark > F Additional results \| **Dataset** \| **Text** \| **Label** \| \| --- \| --- \| --- \| \| AmazonCounterfactualClassification \| In person it looks as though it would ha
- Retrieved 2: score 0.542, p19: MTEB: Massive Text Embedding Benchmark > F Additional results Table 7: STS examples. Scores are continuous between 0 and 5 (included).
- Retrieved 3: score 0.537, p19: MTEB: Massive Text Embedding Benchmark > F Additional results \| **Dataset** \| **First set sentence** \| **Second set sentence** \| \| --- \| --- \| --- \| \| BUCC \| Morales remporte l’élection présidentielle

### paper_09_r0_d10 (factual, paper_09, routed: section_aware)

- Question: What does MTEB stand for?
- Answer: Massive Text Embedding Benchmark
- Evidence p9: we presented the Massive Text Embedding Benchmark (MTEB)
- Evidence p1: models are constantly being proposed without proper evaluation. To solve this problem, we introduce the Massive Text Embedding Benchmark (MTEB). MTEB spans 8 embedding tasks covering a total of 58 datasets and 112 languages.
- Router rank of the first relevant chunk: none in the top 10
- Found in the top 5 by: fixed_size, recursive
- Retrieved 1: score 0.682, p2: MTEB: Massive Text Embedding Benchmark > 2 Related Work > 2.1 Benchmarks MTEB unifies datasets from different embedding tasks into a common, accessible evaluation framework. MTEB incorporates SemEval 
- Retrieved 2: score 0.671, p3-4: MTEB: Massive Text Embedding Benchmark > 3 The MTEB Benchmark > 3.2 Tasks and Evaluation Reranking Inputs are a query and a list of relevant and irrelevant reference texts. The aim is to rank the resu
- Retrieved 3: score 0.668, p4: MTEB: Massive Text Embedding Benchmark > 3 The MTEB Benchmark > 3.2 Tasks and Evaluation ImdbClassification MassiveIntentClassification MassiveScenarioClassification MTOPDomainClassification MTOPInten

### paper_09_r0_d14 (table, paper_09, routed: section_aware)

- Question: What is the average score for the coCondenser-msmarco model?
- Answer: 52.35
- Evidence p5: \| coCondenser-msmarco \| 64.71 \| 37.64 \| 81.74 \| 51.84 \| 32.96 \| 76.47 \| 29.50 \| 52.35 \|
- Evidence p21: 29.91<br>30.08<br><br> \| \| Average \| 41.97 \| 42.06 \| 38.33 \| 45.45 \| 48.72 \| 52.35 \| 56.00<br>4 \| 0.28 \| 45.21 \| 34.95 \| 56. \| 6<br>56.53 \| 52.44 \|
- Router rank of the first relevant chunk: none in the top 10
- Found in the top 5 by: fixed_size
- Retrieved 1: score 0.685, p5-6: MTEB: Massive Text Embedding Benchmark > 4 Results > 4.1 Models We evaluate on the test splits of all datasets except for MSMARCO, where the dev split is used following Thakur et al. (2021). We benchm
- Retrieved 2: score 0.684, p21: MTEB: Massive Text Embedding Benchmark > F Additional results \| MSMARCO \| 9.60 \| 9.75 \| 1.91 \| 9.35 \| 11.00 \| 29.57 \| 36.77<br>5 \| .58 \| 7.60 \| 1.09 \| 36. \| 54<br>39.03 \| 23.72 \| 39.75 \| 26.60 \| 10.53
- Retrieved 3: score 0.661, p8: MTEB: Massive Text Embedding Benchmark > 4 Results > 4.2 Analysis Retrieval SGPT-5.8B-msmarco is the best embedding model on the BEIR subset in MTEB as well as on the full BEIR benchmark (Thakur et al

