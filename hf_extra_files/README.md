---
base_model:
- Qwen/Qwen2.5-VL-7B-Instruct
language:
- en
metrics:
- accuracy
pipeline_tag: zero-shot-object-detection
library_name: transformers
---

<div align=center>
  <img src="assets/logo.png" width=300 >
</div>

# 🦖🧠 Rex-Thinker: Grounded Object Refering via Chain-of-Thought Reasoning 🦖🧠

<div align=center>

<p align="center">
  <a href="https://bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/RexThinker-Website-Red?logo=afdian&logoColor=white&color=blue"
      alt="RexThinker Website"
    />
  </a>
  <a href="https://github.com/IDEA-Research/Rex-Thinker/blob/master/paper_temp/rexthinker.pdf">
    <img
      src="https://img.shields.io/badge/RexThinker-Paper-Red%25red?logo=arxiv&logoColor=red&color=yellow
"
      alt="RexThinker Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT">
    <img 
        src="https://img.shields.io/badge/RexThinker-Weight-orange?logo=huggingface&logoColor=yellow" 
        alt="RexThinker weight on Hugging Face"
    />
  </a>
  <a href="https://demo.bagel-ai.org/">
    <img
      src="https://img.shields.io/badge/RexThinker-Data-orange?logo=huggingface&logoColor=yellow" 
      alt="RexThinker data on Hugging Face"
    />
  </a>
  
</p>

</div>

> We propose Rex-Thinker, a Chain-of-Thought (CoT) reasoning model for object referring that addresses two key challenges: lack of interpretability and inability to reject unmatched expressions. Instead of directly predicting bounding boxes, Rex-Thinker reasons step-by-step over candidate objects to determine which, if any, match a given expression. Rex-Thinker is trained in two stages: supervised fine-tuning to learn structured CoT reasoning, followed by reinforcement learning with GRPO to enhance accuracy, faithfulness, and generalization. Our approach improves both prediction precision and interpretability, while enabling the model to abstain when no suitable object is found. Below is an example of the model's reasoning process:

<p align="center"><img src="assets/teaser_example.jpg" width="95%"></p>


## Method

**Rex-Thinker** reformulates object referring as a **Chain-of-Thought (CoT)** reasoning task to improve both interpretability and reliability. The model follows a structured three-stage reasoning paradigm:

1. **Planning**: Decompose the referring expression into interpretable subgoals.

2. **Action**: Evaluate each candidate object (obtained via an open-vocabulary detector) against these subgoals using step-by-step reasoning.

3. **Summarization**: Aggregate the intermediate results to output the final prediction — or abstain when no object matches.

Each reasoning step is grounded in a specific candidate object region through **Box Hints**, making the process transparent and verifiable.

Rex-Thinker is implemented on top of **QwenVL-2.5**, and trained in two stages:

- **Supervised Fine-Tuning (SFT)**  
  Cold-start training using GPT-4o-generated CoT traces as supervision.

- **GRPO-based Reinforcement Learning**  
  Further optimizes reasoning accuracy, generalization, and rejection ability via Group Relative Policy Optimization.

This CoT-based framework enables Rex-Thinker to make faithful, interpretable predictions while generalizing well to out-of-domain referring scenarios.


<p align="center"><img src="assets/model.jpg" width="95%"></p>



## 1. Installation ⛳️

```bash
conda create -n rexthinker -m python=3.10
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -v -e .

# additional packages Grounding DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
##  To support torch2.6
git remote add quantumope https://github.com/QuantuMope/GroundingDINO.git
git fetch quantumope PR/andrew/add-torch26-support-ms-deform-attn
git merge quantumope/PR/andrew/add-torch26-support-ms-deform-attn
##  Continue with installation
pip install -v -e .
mkdir weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights
cd ..
```

### 1.1 Download Pre-trained Model
We provide the pre-trained model weights of Rex-Thinker-GRPO, which is trained on HumanRef-CoT through SFT and GRPO. You can download the model weights from [Hugging Face](https://huggingface.co/IDEA-Research/Rex-Thinker-GRPO-7B).

Or you can also using the following command to download the pre-trained models:
```bash
git lfs install
git clone https://huggingface.co/IDEA-Research/Rex-Thinker-GRPO-7B IDEA-Research/Rex-Thinker-GRPO-7B
```

## 2. Inference 🚀
We provide a simple inference script to test the model. In this script, we use Grouning DINO to get the candidate boxes.  You can run the following command to test the model:

```bash
CUDA_VISIBLE_DEVICES=0 python demo/inference_single_image.py \
  --image_path demo/example_images/demo_helmet.png \
  --cate_name helmet \
  --ref_exp the forth helmet from left \
  --vis_path vis/example_output.jpg 
```

You will get output fromt the terminal like this:
```text
<think>OK, the user needs us to detect the fourth helmet from left. To accomplish this task, I need to break it down into the following steps:
- Step 1: Sort the helmets from left to right.
- Step 2: Find the fourth helmet from the sorted list.

# Step 1: Sort the helmets from left to right
I see 6 helmets in this image, and their order from left to right is [Helmet 5, Helmet 1, Helmet 3, Helmet 2, Helmet 4, Helmet 6].

# Step 2: Find the fourth helmet from the sorted list
From the sorted list [Helmet 5, Helmet 1, Helmet 3, Helmet 2, Helmet 4, Helmet 6], the fourth helmet from the left is Helmet 2.

# Summarize and Re-Check answer
Let’s now recheck our answer and put ✅ for the target helmet and ❌ for others
- Helmet 5: It is the first helmet from left → ❌
- Helmet 1: It is the second helmet from left → ❌
- Helmet 3: It is the third helmet from left → ❌
- Helmet 2: It is the fourth helmet from left → ✅
- Helmet 4: It is the fifth helmet from left → ❌
- Helmet 6: It is the sixth helmet from left → ❌</think><answer>json
[{"bbox_2d": [578, 359, 825, 580], "label": "the forth helmet from left"}]
</answer>
```

and visulized results like this:
<p align="center"><img src="demo/example_images/demo_output.jpg" width="80%"></p>


## 3. Gradio Demo 🤗
We provide a Gradio demo for you to test the model. You can run the following command to start the Gradio demo:
```bash
CUDA_VISIBLE_DEVICES=0 python demo/gradio_demo.py \
  --model_path IDEA-Research/Rex-Thinker-GRPO-7B \
  --server_ip 0.0.0.0 \
  --server_port 7860
```

Then you can open your browser and visit `http://localhost:7860` to see the Gradio demo. You can input the image path, category name, and referring expression to test the model.

<p align="center"><img src="assets/gradio.jpg" width="95%"></p>

## Citation 📜
```text
@misc{jiang2025rexthinkergroundedobjectreferring,
      title={Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning}, 
      author={Qing Jiang and Xingyu Chen and Zhaoyang Zeng and Junzhi Yu and Lei Zhang},
      year={2025},
      eprint={2506.04034},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.04034}, 
}
```