# HybridSN-small + Spectral QNN Gated Fusion + Prototype Loss

`loss = CE(logits, y) + lambda * CE(-distance(fused_feature, class_prototypes), y)`

|   shot |   HybridSN-small OA |   Proto QNN OA |   Delta OA |   HybridSN-small Macro-F1 |   Proto QNN Macro-F1 |   Delta Macro-F1 |   runs |
|-------:|--------------------:|---------------:|-----------:|--------------------------:|---------------------:|-----------------:|-------:|
|      5 |               72.02 |          28.52 |      -43.5 |                     63.81 |                20.68 |           -43.13 |      1 |
