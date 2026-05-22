# HybridSN-small + Spectral QNN Gated Fusion + Prototype Loss

`loss = CE(logits, y) + lambda * CE(-distance(fused_feature, class_prototypes), y)`

|   shot |   HybridSN-small OA |   Proto QNN OA |   Delta OA |   HybridSN-small Macro-F1 |   Proto QNN Macro-F1 |   Delta Macro-F1 |   runs |
|-------:|--------------------:|---------------:|-----------:|--------------------------:|---------------------:|-----------------:|-------:|
|     10 |               80.12 |          80.88 |       0.76 |                     71.53 |                71.82 |             0.29 |      5 |
