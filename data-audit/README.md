## Data Summary

``CTRL``, ``GPT2`` and ``DoHA`` folders contain manual annotations of knowledge-grounded responses generated by three models trained on three bechmarks:  [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/), 
[CMU-DoG](https://aclanthology.org/D18-1076/?ref=https://githubhelp.com) and [TopicalChat](https://github.com/alexa/Topical-Chat).

The annotations of the gold responses from the three benchmarks are under the folder ``gold``.

Each annotated data contains 200 examples that are stored under a csv file.  

## Data Fields
 - `knowledge`:  The source knowkedge.
 - `history`: The previous utterance in history.
 - `model response`: The model response (e.g., GPT2, DoHA, CTRL).
 - `BEGIN`:  The BEGIN labels for the response.
 - `VRM`:   The VRM labels for the response.
