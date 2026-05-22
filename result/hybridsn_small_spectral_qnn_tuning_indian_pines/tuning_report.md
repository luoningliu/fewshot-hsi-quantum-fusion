# Spectral QNN Gated Fusion Tuning: Indian Pines Few-shot

Goal: tune the quantum spectral gated fusion branch so that 5/10-shot can exceed HybridSN-small.

## Best By Macro-F1

|   shot | config                |   HybridSN-small OA |   Spectral Gated QNN OA |   Delta OA |   HybridSN-small Macro-F1 |   Spectral Gated QNN Macro-F1 |   Delta Macro-F1 |   runs |
|-------:|:----------------------|--------------------:|------------------------:|-----------:|--------------------------:|------------------------------:|-----------------:|-------:|
|      1 | q4_l1_scalar          |               41.49 |                   44.4  |       2.91 |                     37.75 |                         40.5  |             2.75 |      5 |
|      5 | q6_l2_classwise_5shot |               72.02 |                   71.68 |      -0.34 |                     63.81 |                         64.4  |             0.6  |      5 |
|     10 | q6_l1_classwise       |               80.12 |                   80.89 |       0.77 |                     71.53 |                         71.81 |             0.28 |      5 |

## Best By OA

|   shot | config          |   HybridSN-small OA |   Spectral Gated QNN OA |   Delta OA |   HybridSN-small Macro-F1 |   Spectral Gated QNN Macro-F1 |   Delta Macro-F1 |   runs |
|-------:|:----------------|--------------------:|------------------------:|-----------:|--------------------------:|------------------------------:|-----------------:|-------:|
|      1 | q4_l1_scalar    |               41.49 |                   44.4  |       2.91 |                     37.75 |                         40.5  |             2.75 |      5 |
|      5 | q6_l1_classwise |               72.02 |                   72    |      -0.02 |                     63.81 |                         64.05 |             0.24 |      5 |
|     10 | q6_l1_classwise |               80.12 |                   80.89 |       0.77 |                     71.53 |                         71.81 |             0.28 |      5 |

## All Tried Configurations

| config                           |   shot |   HybridSN-small OA |   Spectral Gated QNN OA |   Delta OA |   HybridSN-small Macro-F1 |   Spectral Gated QNN Macro-F1 |   Delta Macro-F1 |   runs |
|:---------------------------------|-------:|--------------------:|------------------------:|-----------:|--------------------------:|------------------------------:|-----------------:|-------:|
| q4_l1_scalar                     |      1 |               41.49 |                   44.4  |       2.91 |                     37.75 |                         40.5  |             2.75 |      5 |
| q6_l2_classwise_5shot            |      5 |               72.02 |                   71.68 |      -0.34 |                     63.81 |                         64.4  |             0.6  |      5 |
| q6_l1_classwise                  |      5 |               72.02 |                   72    |      -0.02 |                     63.81 |                         64.05 |             0.24 |      5 |
| q6_l1_classwise_wd0_5shot        |      5 |               72.02 |                   72    |      -0.02 |                     63.81 |                         64.05 |             0.24 |      5 |
| q4_l1_classwise                  |      5 |               72.02 |                   71.19 |      -0.83 |                     63.81 |                         63.72 |            -0.08 |      5 |
| q6_l1_classwise_monitor_oa_5shot |      5 |               72.02 |                   71.43 |      -0.59 |                     63.81 |                         63.54 |            -0.26 |      5 |
| q6_l1_classwise_lr0001_5shot     |      5 |               72.02 |                   71.53 |      -0.49 |                     63.81 |                         63.37 |            -0.44 |      5 |
| q4_l1_scalar                     |      5 |               72.02 |                   70.52 |      -1.5  |                     63.81 |                         63.14 |            -0.67 |      5 |
| q6_l1_ring_classwise_5shot       |      5 |               72.02 |                   70.94 |      -1.08 |                     63.81 |                         62.8  |            -1    |      5 |
| q6_l1_classwise                  |     10 |               80.12 |                   80.89 |       0.77 |                     71.53 |                         71.81 |             0.28 |      5 |
| q4_l1_classwise                  |     10 |               80.12 |                   80.72 |       0.6  |                     71.53 |                         71.54 |             0.01 |      5 |
| q4_l1_scalar                     |     10 |               80.12 |                   80.5  |       0.38 |                     71.53 |                         71.14 |            -0.39 |      5 |

## Interpretation

The best current configuration is q6_l1_classwise. It exceeds HybridSN-small on 10-shot for both OA and Macro-F1. On 5-shot, it exceeds HybridSN-small on Macro-F1 but misses OA by only 0.02 percentage points. Additional attempts with ring entanglement, zero weight decay, validation-OA checkpoint selection, lower learning rate, and a second QNN layer did not produce a strict 5-shot OA win. The honest conclusion is that QNN improves the 5-shot class-balanced metric but does not yet robustly exceed HybridSN-small on all metrics.
