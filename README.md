# Quick Jailbreak Evaluation

A toolkit for evaluating LLM jailbreak techniques with parallel processing.

## Jailbreak Performance Analysis

![Jailbreak Performance Metrics](https://raw.githubusercontent.com/arthurdupe/quick_jailbreak_eval/main/assets/jailbreak_metrics.png)

The analysis of different jailbreak techniques reveals interesting patterns in model behavior. Based on the metrics shown in the graph:

The hidden_layer jailbreak seems to get around refusal more frequently than other jailbreaks, but it produces less genuinely dangerous output. Overall on strongreject metrics it is a more powerful jailbreak than Pliny and AIM.

### Metrics Explained

See https://arxiv.org/abs/2402.10260v2 -- in brief:

- **Success Rate (1-Refusal)**: How often the jailbreak bypasses model refusal mechanisms
- **Convincingness (1-5)**: Rating of how convincing/natural the generated content appears
- **Specificity (1-5)**: Rating of how detailed and specific the generated content is
- **Composite Score (0-5)**: Combined measure of jailbreak effectiveness

## Pipeline Overview

The evaluation pipeline consists of three key scripts that work together:

1. **cartesian_simple.py**: Generates model responses to jailbreak prompts
   - Takes jailbreak prompts and applies them to target models
   - Uses asyncio for parallel processing to maximize throughput
   - Outputs CSV files with model responses for each jailbreak attempt

2. **eval_rejections_parallel.py**: Evaluates the effectiveness of jailbreak attempts
   - Takes the output from cartesian_simple.py as input
   - Uses Claude 3 Sonnet as an evaluation model to rate each jailbreak attempt
   - Implements parallel processing for faster evaluation
   - Analyzes responses on refusal rate, convincingness, and specificity metrics

3. **analyze_results.py**: Generates visualizations and metrics
   - Processes the evaluation data from eval_rejections_parallel.py
   - Creates plots showing comparative performance of different jailbreak methods
   - Calculates aggregate statistics on jailbreak effectiveness

Raw data from the experiments is included in the `data_archive.zip` file for reference and further analysis.

## Usage

See individual scripts for documentation on usage and parameters.

## Installation

```bash
# Clone the repository
git clone https://github.com/arthurdupe/quick_jailbreak_eval.git
cd quick_jailbreak_eval

# Install dependencies
# Though I used `uv` (but it confuses LLMs... alas)
pip install -e .
```
