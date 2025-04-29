#!/usr/bin/env python
"""Evaluate jailbreak responses using strong_reject prompts with parallel processing."""

import asyncio
import pandas as pd
import os
import sys
import time
import argparse
from pathlib import Path
import dotenv
from typing import List, Dict, Any, Optional

# Import safety-tooling components
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

# Import the strong_reject prompt
from prompt_28_apr.strong_reject import MAIN_PROMPT, DEVELOPER_PROMPT

# Load environment variables
dotenv.load_dotenv("/workspace/jailbreaking/.env")

# Set dummy OpenAI API key if not present
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("Setting dummy OPENAI_API_KEY...")
    os.environ["OPENAI_API_KEY"] = "dummy-key-not-used"

# Add Anthropic API key directly for testing
os.environ["ANTHROPIC_API_KEY"] = "dummy-key-not-used"

# Check OpenRouter API key
if "OPENROUTER_API_KEY" in os.environ and os.environ["OPENROUTER_API_KEY"]:
    print(f"Found OPENROUTER_API_KEY: {os.environ['OPENROUTER_API_KEY'][:5]}...{os.environ['OPENROUTER_API_KEY'][-5:]}")
else:
    print("WARNING: OPENROUTER_API_KEY not found in environment variables!")

# Configure the API
api = InferenceAPI(
    cache_dir=Path(os.path.join(os.getcwd(), "prompt_28_apr", "prompt_history")),
)

async def evaluate_sample(sample, eval_model, sample_num, total_samples):
    """Evaluate a single jailbreak sample with the specified model."""
    prompt = sample['prompt']
    response = sample['response']
    jailbreak = sample['jailbreak']
    
    # Format the strong reject prompt
    formatted_prompt = MAIN_PROMPT.format(
        forbidden_prompt=prompt,
        response=response
    )
    
    # Create a prompt object with the formatted prompt, including system message
    evaluation_prompt = Prompt(messages=[
        ChatMessage(content=DEVELOPER_PROMPT, role=MessageRole.system),
        ChatMessage(content=formatted_prompt, role=MessageRole.user)
    ])
    
    print(f"Evaluating sample {sample_num}/{total_samples}: {jailbreak} - {prompt[:30]}...")
    
    try:
        # Call the evaluation model via OpenRouter
        eval_response = await api(
            model_id="anthropic/claude-3.7-sonnet",
            prompt=evaluation_prompt,
            max_attempts_per_api_call=1,
            force_provider="openrouter"
        )
        
        # Extract the completion
        try:
            if hasattr(eval_response, 'completions') and eval_response.completions:
                completion = eval_response.completions[0].text
            elif hasattr(eval_response, 'completion'):
                completion = eval_response.completion
            elif isinstance(eval_response, list) and hasattr(eval_response[0], 'completion'):
                completion = eval_response[0].completion
            elif isinstance(eval_response, dict) and 'content' in eval_response:
                completion = eval_response['content']
            elif isinstance(eval_response, dict) and 'text' in eval_response:
                completion = eval_response['text']
            elif hasattr(eval_response, 'content'):
                completion = eval_response.content  
            else:
                completion = str(eval_response)
            print(f"Successfully extracted completion: {completion[:100]}...")
        except Exception as extract_error:
            print(f"Error extracting completion: {extract_error}")
            completion = "Error extracting response"
        
        return {
            "prompt": prompt,
            "response": response,
            "jailbreak": jailbreak,
            "evaluation": completion,
            "timestamp": int(time.time()),
            "model": eval_model,
        }
    except Exception as e:
        print(f"Error processing sample: {e}")
        return {
            "prompt": prompt,
            "response": response,
            "jailbreak": jailbreak,
            "evaluation": f"ERROR: {str(e)}",
            "timestamp": int(time.time()),
            "model": eval_model,
        }

async def process_evaluation_batch(batch, eval_model, batch_idx, total_batches, batch_size):
    """Process a batch of evaluations in parallel."""
    print(f"Processing batch {batch_idx}/{total_batches}")
    batch_start_time = time.time()
    
    total_samples = sum(len(b) for b in [batch for _ in range(total_batches)])
    
    # Create concurrent tasks for all samples in the batch
    tasks = []
    for sample_idx, sample in enumerate(batch, 1):
        sample_num = (batch_idx-1) * batch_size + sample_idx
        task = evaluate_sample(sample, eval_model, sample_num, total_samples)
        tasks.append(task)
    
    # Execute all sample evaluations concurrently
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    results = []
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            print(f"Error processing sample in batch {batch_idx}: {result}")
            # Create an error record
            error_sample = batch[i]
            results.append({
                "prompt": error_sample.get('prompt', 'Unknown'),
                "response": error_sample.get('response', 'Unknown'),
                "jailbreak": error_sample.get('jailbreak', 'Unknown'),
                "evaluation": f"ERROR: {str(result)}",
                "timestamp": int(time.time()),
                "model": eval_model,
            })
        else:
            results.append(result)
    
    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    print(f"Batch {batch_idx} completed in {batch_duration:.2f} seconds")
    
    return results

async def run_evaluations(
    input_file,
    output_file=None,
    eval_model="anthropic/claude-3.7-sonnet",
    num_samples=None,
    batch_size=20,
    max_concurrent_batches=5
):
    """Run evaluations on jailbreak samples with parallel processing."""
    # Generate timestamp for file naming
    timestamp = int(time.time())
    
    # Ensure output directory exists
    output_dir = Path("/workspace/jailbreaking/data/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_file is None:
        # Extract model name for the filename (remove special chars)
        model_name_clean = eval_model.replace('/', '_').replace('.', '_').replace('-', '_')
        output_file = output_dir / f"evaluations_{model_name_clean}_{timestamp}.csv"
    
    # Load the input data
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} samples from {input_file}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()
    
    # Limit samples if specified
    if num_samples is not None and num_samples < len(df):
        df = df.head(num_samples)
        print(f"Limited to first {num_samples} samples")
    
    # Create batches
    samples = df.to_dict('records')
    batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
    print(f"Divided {len(samples)} samples into {len(batches)} batches of size {batch_size}")
    
    # Create all batch processing tasks
    print(f"Creating {len(batches)} batch processing tasks for parallel execution...")
    batch_tasks = []
    for i, batch in enumerate(batches, 1):
        task = process_evaluation_batch(batch, eval_model, i, len(batches), batch_size)
        batch_tasks.append(task)
    
    # Process batches in groups to respect rate limiting
    all_results = []
    for i in range(0, len(batch_tasks), max_concurrent_batches):
        current_batch_tasks = batch_tasks[i:i+max_concurrent_batches]
        group_num = i//max_concurrent_batches + 1
        total_groups = (len(batch_tasks)-1)//max_concurrent_batches + 1
        print(f"Executing batch group {group_num}/{total_groups} with {len(current_batch_tasks)} batches...")
        
        start_time = time.time()
        current_results = await asyncio.gather(*current_batch_tasks)
        end_time = time.time()
        
        print(f"Batch group {group_num} completed in {end_time - start_time:.2f} seconds")
        
        # Flatten results from all batches in this group
        for batch_result in current_results:
            all_results.extend(batch_result)
            
        # Save intermediate results after each group of batches
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_file, index=False)
        print(f"Saved intermediate results to {output_file} ({len(all_results)} samples processed)")
    
    # Create final dataframe
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_file, index=False)
    print(f"Saved final results to {output_file}")
    
    return result_df

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate jailbreak responses with parallel processing')
    parser.add_argument('--input_file', type=str, 
                      default="/workspace/jailbreaking/data/sample_formatted.csv",
                      help='Input CSV file with jailbreak results')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output CSV file for evaluation results')
    parser.add_argument('--eval_model', type=str, default="anthropic/claude-3.7-sonnet",
                      help='Model to use for evaluations')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to evaluate (None to evaluate all samples)')
    parser.add_argument('--batch_size', type=int, default=20,
                      help='Number of samples per batch')
    parser.add_argument('--max_concurrent_batches', type=int, default=5,
                      help='Maximum number of batches to process concurrently')
    
    # Parse arguments, checking if running from command line or imported
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Default values for testing when imported
        args = parser.parse_args([])
        
    print(f"Using evaluation model: {args.eval_model}")
    print(f"Input file: {args.input_file}")
    if args.num_samples:
        print(f"Processing {args.num_samples} samples")
    else:
        print("Processing all available samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Max concurrent batches: {args.max_concurrent_batches}")
    
    # Run evaluations
    return await run_evaluations(
        input_file=args.input_file,
        output_file=args.output_file,
        eval_model=args.eval_model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_concurrent_batches=args.max_concurrent_batches
    )

if __name__ == "__main__":
    asyncio.run(main())