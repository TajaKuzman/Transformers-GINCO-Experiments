## The Results of the Transformer Comparison Based on Language

### Max_seq_length = 300, Batches: 32

1. SloBERTa:
* 1st run: Macro f1: 0.531, Micro f1: 0.619
* 2nd run: Macro f1: 0.592, Micro f1: 0.624
* 3rd run: Macro f1: 0.572, Micro f1: 0.594
* 4th run: Macro f1: 0.522, Micro f1: 0.599
* 5th run: Macro f1: 0.539, Micro f1: 0.609

2. CroSloEngual BERT:
* 1st run: Macro f1: 0.374, Micro f1: 0.518
* 2nd run: Macro f1: 0.454, Micro f1: 0.508
* 3rd run: Macro f1: 0.379, Micro f1: 0.503
* 4th run: Macro f1: 0.388, Micro f1: 0.497
* 5th run: Macro f1: 0.462, Micro f1: 0.548

3. XML-RoBERTa Base:
* 1st run: Macro f1: 0.518, Micro f1: 0.523
* 2nd run: Macro f1: 0.557, Micro f1: 0.558
* 3rd run: Macro f1: 0.57, Micro f1: 0.563
* 4th run: Macro f1: 0.559, Micro f1: 0.569
* 5th run: Macro f1: 0.556, Micro f1: 0.558

4. mDeBERTaV3 (??? - error, assigning all instances to one class (Research Article, Invitation) Macro f1: 0.00327, Micro f1: 0.0355)

5. BERTić:
* 1st run: Macro f1: 0.379, Micro f1: 0.472
* 2nd run: Macro f1: 0.488, Micro f1: 0.492
* 3rd run: Macro f1: 0.453, Micro f1: 0.492
* 4th run: Macro f1: 0.482, Micro f1: 0.492
* 5th run: Macro f1: 0.472, Micro f1: 0.482

6. BERT:
* 1st run: Macro f1: 0.164, Micro f1: 0.274
* 2nd run: Macro f1: 0.27, Micro f1: 0.294
* 3rd run: Macro f1: 0.262, Micro f1: 0.299
* 4th run: Macro f1: 0.276, Micro f1: 0.33
* 5th run: Macro f1: 0.23, Micro f1: 0.289

### Max_seq_length = 128, batches = 21

1. SloBERTa
* 1st run: Macro f1: 0.48, Micro f1: 0.553
* 2nd run: Macro f1: 0.506, Micro f1: 0.574
* 3rd run: Macro f1: 0.573, Micro f1: 0.574
* 4th run: Macro f1: 0.565, Micro f1: 0.584
* 5th run: Macro f1: 0.516, Micro f1: 0.563

2. CroSloEngualBERT
* 1st run: Macro f1: 0.389, Micro f1: 0.497
* 2nd run: Macro f1: 0.426, Micro f1: 0.513
* 3rd run: Macro f1: 0.441, Micro f1: 0.513
* 4th run: Macro f1: 0.445, Micro f1: 0.518
* 5th run: Macro f1: 0.441, Micro f1: 0.518

3. XMLRoBERTa Large - The training on XMLRoBERTa Large takes at least twice more time.
* 1st run: Macro f1: 0.6, Micro f1: 0.619
* 2nd run: Macro f1: 0.589, Micro f1: 0.594
* 3rd run: Macro f1: 0.54, Micro f1: 0.548
* 4th run: Macro f1: 0.507, Micro f1: 0.579
* 5th run: Macro f1: 0.546, Micro f1: 0.553

4. XMLRoBERTa Base:
* 1st run: Macro f1: 0.41, Micro f1: 0.477
* 2nd run: Macro f1: 0.438, Micro f1: 0.513
* 3rd run: Macro f1: 0.458, Micro f1: 0.503
* 4th run: Macro f1: 0.448, Micro f1: 0.497
* 5th run: Macro f1: 0.393, Micro f1: 0.442

5. BERTić:
* 1st run: Macro f1: 0.378, Micro f1: 0.467
* 2nd run: Macro f1: 0.391, Micro f1: 0.462
* 3rd run: Macro f1: 0.399, Micro f1: 0.497
* 4th run: Macro f1: 0.393, Micro f1: 0.472
* 5th run: Macro f1: 0.374, Micro f1: 0.462

6. BERT:
* 1st run: Macro f1: 0.2, Micro f1: 0.284
* 2nd run: Macro f1: 0.182, Micro f1: 0.249
* 3rd run: Macro f1: 0.193, Micro f1: 0.249
* 4th run: Macro f1: 0.22, Micro f1: 0.239
* 5th run: Macro f1: 0.219, Micro f1: 0.259