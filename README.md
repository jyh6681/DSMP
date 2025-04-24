# DSMP
This is a repository for the JMLR paper [Limiting Over-Smoothing and Over-Squashing of Graph Message Passing by Deep Scattering Transforms](https://arxiv.org/abs/2407.06988)
## Main Contribution
We propose the DSMP model in this paper, which is a discriminatively trained, multi-layer graph neural network model designed to overcome over-smoothing, over-squashing and instabilty issues.

By harnessing spectral transformation, the
DSMP model aggregates neighboring nodes with global information, thereby enhancing
the precision and accuracy of graph signal processing. We provide theoretical proofs
demonstrating the DSMP’s effectiveness in mitigating these issues under specific conditions.
Additionally, we support our claims with empirical evidence and thorough frequency analysis,
showcasing the DSMP’s superior ability to address instability, over-smoothing, and oversquashing.

## Below is a brief instruction of implementing DSMP model on graph-level classification task.

### For experiments on ogbg-molhiv dataset:

```
python3 smp_noabs_molhiv arg_train # if test for the first time, use arg_train action, else use ray_train for multiple tests.
```
### For experiments on other graph-level task:

```
python3 smp_graphlevel_noabs arg_train --dataname COLLAB # need to specify the name of the dataset
```

### For experiments on node-level task:

```
python3 smp_nodeclass_noabs arg_train --dataname wisconsin # need to specify the name of the dataset
```
