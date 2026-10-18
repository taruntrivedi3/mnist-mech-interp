# MNIST mechanistic interpretability

A learning notebook that trains linear classifiers and small convolutional networks on MNIST, then explores what the trained models use to make predictions.

[Open the notebook](nn_tutorial.ipynb). It contains only code cells; explanations are short code comments. Run all cells in order. The training progression comes from PyTorch's [What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html), by Jeremy Howard / fast.ai, with thanks to Rachel Thomas and Francisco Ingham.

## Run

Use Python 3.11. For a CPU environment:

```bash
git clone https://github.com/taruntrivedi3/mnist-mech-interp.git
cd mnist-mech-interp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
jupyter lab nn_tutorial.ipynb
```

MNIST downloads automatically from the PyTorch tutorial repository; the download is checksum-verified before loading. Data and generated metrics stay in ignored `data/` and `artifacts/` directories. The notebook sets a seed and uses two CPU threads. The tutorial's final accelerator section uses an available accelerator; interpretability runs on CPU copies of the trained models. GPU execution has not been verified.

## Experiments

- **Linear weight maps and exact pixel contributions:** decompose a target-versus-alternative logit difference into pixel contributions plus a bias term, and verify the reconstruction.
- **Internal activations:** inspect first-layer filters, second-layer feature maps, and highly activating training examples. High activation alone does not show causal importance.
- **Channel ablation:** replace each second-layer channel with zero or its training-set mean. Rank channels on 512 training examples and report changes in loss and accuracy on a separate 512-image validation subset.
- **Activation patching:** cover a fixed central region of 128 validation images, then patch clean internal activations into the corrupted forward pass. Compare individual channels, a mismatched-image donor, same-corrupted-activation controls, and full-layer restoration.
- **Input attribution:** visualize gradient saliency and integrated gradients of a fixed class contrast. Check integrated gradients against an analytic linear-model answer and report the CNN's numerical completeness residual.

The original notebook's scratch expressions were repaired so it runs in order. The manual log-softmax uses `logsumexp` for numerical stability. Copies of the trained linear classifier and final CNN are retained before further analysis. The models are trained by the notebook; no pretrained checkpoint is bundled.

## Reading the results

Accuracy and loss use explicitly sized validation subsets, not the untouched test set. Channel selection uses training data. The patching analysis reports results for all 128 images and a recovery fraction restricted to images that were correctly classified before corruption and whose true-versus-fixed-alternative margin dropped by more than 0.1. That fraction is the total recovered margin divided by the total original margin gap on eligible images; its denominator is printed. Negative values and values above one are possible.

The attribution illustration deliberately selects the eligible image most damaged by occlusion. It is an example of a failure, not a representative sample. Patching outcomes depend on the corruption and donor; zero ablation can create unusual internal states. These experiments test model behavior under specific interventions and do not establish a complete digit-recognition circuit or uniquely meaningful neurons.

The notebook includes assertions for exact linear decomposition, analytic linear integrated gradients, no-op hooks, same-corrupted-donor patching, full-layer clean recovery, and unchanged CNN parameters and predictions after intervention. Full-layer recovery is an implementation control: this CNN has no connections that bypass the patched layer.

The committed notebook was executed from a fresh CPU kernel: all 77 code cells and their assertions passed. On the 512-image validation subset, the linear classifier reached 88.9% accuracy and the CNN 92.2%. Central occlusion reduced CNN accuracy on the 128-image patching subset from 93.8% to 25.0%. Channel 11, selected by training-set ablation, recovered 16.4% of the total lost margin across 95 eligible images. The same-channel mismatched donor recovered -3.0%; the full-layer control recovered 100%. These are results from one seed and one corruption, not a robustness benchmark. The CNN integrated-gradients completeness residual was 0.056 against an endpoint contrast change of 5.968 using 128 integration steps.

## Attribution

The training code is adapted from the PyTorch tutorial, including modifications made while working through it. The interpretability experiments extend that training workflow. The upstream BSD 3-Clause license is retained in [LICENSE](LICENSE), sourced from [pytorch/tutorials](https://github.com/pytorch/tutorials/blob/main/LICENSE).
