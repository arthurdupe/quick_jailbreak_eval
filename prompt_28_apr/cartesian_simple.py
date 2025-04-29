"""Run cartesian product of jailbreaks and prompts with error handling."""

import asyncio
import pandas as pd
import time
import os
import sys
from typing import List, Dict, Any, Optional
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
import datetime

# Import jailbreaks from the local module
from prompt_28_apr.jailbreaks import JAILBREAKS

# Configure the API
from pathlib import Path

# Set API keys from .env file
import dotenv
dotenv.load_dotenv("/workspace/jailbreaking/.env")

# Set dummy OpenAI API key if not present
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("Setting dummy OPENAI_API_KEY...")
    os.environ["OPENAI_API_KEY"] = "dummy-key-not-used"

# Check OpenRouter API key
if "OPENROUTER_API_KEY" in os.environ and os.environ["OPENROUTER_API_KEY"]:
    print(f"Found OPENROUTER_API_KEY: {os.environ['OPENROUTER_API_KEY'][:5]}...{os.environ['OPENROUTER_API_KEY'][-5:]}")
else:
    print("WARNING: OPENROUTER_API_KEY not found in environment variables!")

api = InferenceAPI(
    cache_dir=Path(os.path.join(os.getcwd(), "prompt_28_apr", "prompt_history")),
)

async def process_batch(batch_items, batch_num, total_batches):
    """Process a batch of items with error handling."""
    print(f"Running batch {batch_num}/{total_batches} with {len(batch_items)} items")
    batch_start_time = time.time()
    
    results = []
    for item in batch_items:
        try:
            jailbreak_name = item["jailbreak_name"]
            jailbreak_fn = item["jailbreak_fn"]
            prompt = item["prompt"]
            
            # Apply the jailbreak function to the prompt
            jailbroken_content = jailbreak_fn(prompt)
            
            # Create a Prompt object with ChatMessage
            jailbroken_prompt = Prompt(messages=[ChatMessage(content=jailbroken_content, role=MessageRole.user)])
            
            print(f"Sending request for: {jailbreak_name} - {prompt[:30]}...")
            
            # Make the actual API call to the model
            response = await api(
                model_id="google/gemini-flash-1.5-8b",
                prompt=jailbroken_prompt,
                max_attempts_per_api_call=1,
                force_provider="openrouter",  # Use OpenRouter for wider model support
            )
            
            # Extract the completion from the response (handle different output formats)
            try:
                if hasattr(response, 'completions') and response.completions:
                    completion = response.completions[0].text
                elif hasattr(response, 'completion'):
                    completion = response.completion
                elif hasattr(response[0], 'completion'):
                    completion = response[0].completion
                else:
                    completion = str(response)
            except Exception as extract_error:
                print(f"Error extracting completion: {extract_error}")
                completion = "Error extracting response"
            
            # Add timestamp to the results
            current_time = int(time.time())
            results.append({
                "prompt": prompt,
                "jailbreak": jailbreak_name,
                "jailbroken_prompt": jailbroken_content,  # Store only the content string
                "response": completion,
                "timestamp": current_time
            })
            
        except Exception as e:
            print(f"Error processing item: {e}")
            current_time = int(time.time())
            results.append({
                "prompt": item["prompt"],
                "jailbreak": item["jailbreak_name"],
                "jailbroken_prompt": f"ERROR: {str(e)}",
                "response": f"ERROR: {str(e)}",
                "timestamp": current_time
            })
    
    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    print(f"Batch {batch_num} completed in {batch_duration:.2f} seconds")
    
    return results

def ensure_data_dir_exists():
    """Ensure the data directory exists."""
    data_dir = "/workspace/jailbreaking/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

async def run_jailbreak_prompt_cartesian_product(
    max_prompts: Optional[int] = 10,
    max_jailbreaks: Optional[int] = 3,
    dataframe: pd.DataFrame = None,
    model_name: str = "google/gemini-flash-1.5-8b",
    batch_size: int = 20,
) -> pd.DataFrame:
    """
    Run a cartesian product of jailbreaks and prompts with improved error handling.
    """
    # Generate timestamp for file naming
    timestamp = int(time.time())
    
    # Ensure data directory exists
    data_dir = ensure_data_dir_exists()
    
    # Load data from parquet if dataframe not provided
    if dataframe is None:
        try:
            dataframe = pd.read_parquet("/workspace/jailbreaking/data/train-00000-of-00001.parquet")
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return pd.DataFrame()
    
    # Get all prompts
    prompts = dataframe["prompt"].tolist()  # Using the actual column name 'prompt'
    print(f"Total prompts in dataset: {len(prompts)}")
    
    # Limit prompts for testing if max_prompts is specified
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
        print(f"Limited to first {max_prompts} prompts")
    
    # Get all jailbreak methods
    jailbreak_methods = list(JAILBREAKS.items())
    print(f"Total jailbreak methods: {len(jailbreak_methods)}")
    
    # Limit jailbreak methods if max_jailbreaks is specified
    if max_jailbreaks is not None:
        jailbreak_methods = jailbreak_methods[:max_jailbreaks]
        print(f"Limited to first {max_jailbreaks} jailbreak methods")
    
    print(f"Using {len(prompts)} prompts from parquet data")
    print(f"Using {len(jailbreak_methods)} jailbreak methods")
    
    # Create all combinations
    items = []
    for prompt in prompts:
        for jailbreak_name, jailbreak_fn in jailbreak_methods:
            items.append({
                "prompt": prompt,
                "jailbreak_name": jailbreak_name,
                "jailbreak_fn": jailbreak_fn,
            })
    
    print(f"Creating {len(items)} total combinations")
    
    # Use the batch_size parameter (default 20 or overridden by argument)
    
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    print(f"Running {len(items)} API calls in {len(batches)} batches of size {batch_size}")
    
    # Create all batch processing tasks at once
    print("Creating batch processing tasks for parallel execution...")
    batch_tasks = []
    for i, batch in enumerate(batches, 1):
        task = process_batch(batch, i, len(batches))
        batch_tasks.append(task)
    
    # Execute all batch tasks concurrently
    print(f"Executing {len(batch_tasks)} batch tasks in parallel...")
    start_time = time.time()
    all_batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    end_time = time.time()
    
    print(f"All batches completed in {end_time - start_time:.2f} seconds")
    
    # Process all results
    all_results = []
    for i, batch_result in enumerate(all_batch_results, 1):
        if isinstance(batch_result, Exception):
            print(f"Error processing batch {i}: {batch_result}")
            continue
            
        all_results.extend(batch_result)
        
        # Save intermediate results
        if i % 5 == 0 or i == len(all_batch_results):
            df = pd.DataFrame(all_results)
            # Extract model name for the filename (remove special chars)
            model_name_clean = model_name.replace('/', '_').replace('.', '_').replace('-', '_')
            batch_file_path = os.path.join(data_dir, "parquets", f"jailbreak_results_{model_name_clean}_batch_{i}_{timestamp}.csv")
            df.to_csv(batch_file_path, index=False)
            print(f"Saved intermediate results to {batch_file_path}")
    
    # Create final dataframe
    result_df = pd.DataFrame(all_results)
    # Extract model name for the filename (remove special chars)
    model_name_clean = model_name.replace('/', '_').replace('.', '_').replace('-', '_')
    final_file_path = os.path.join(data_dir, f"jailbreak_results_{model_name_clean}_{timestamp}.csv")
    result_df.to_csv(final_file_path, index=False)
    print(f"Saved final results to {final_file_path}")
    return result_df

async def main():
    # Parse command line arguments
    import argparse
    
    # Helper function to handle None values
    def parse_optional_int(value):
        if value.lower() == 'none':
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Expected int or 'None', got {value}")
    
    parser = argparse.ArgumentParser(description='Run jailbreak prompt cartesian product')
    parser.add_argument('--model_name', type=str, default="google/gemini-flash-1.5-8b", 
                        help='Model name to use for testing')
    parser.add_argument('--max_prompts', type=parse_optional_int, default=10, 
                        help='Maximum number of prompts to test (use "None" to test all prompts)')
    parser.add_argument('--max_jailbreaks', type=parse_optional_int, default=2, 
                        help='Maximum number of jailbreak methods to test (use "None" to test all methods)')
    parser.add_argument('--requests_per_minute', type=int, default=10000,
                        help='Number of requests to allow per minute (rate limiting)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Number of requests to batch together')
    
    args = parser.parse_args()
    print(f"Using model: {args.model_name}")
    
    # Update the process_batch function to use the specified model
    global process_batch
    original_process_batch = process_batch
    
    async def process_batch_with_model(batch_items, batch_num, total_batches):
        """Wrapper around process_batch that uses the specified model."""
        nonlocal args
        
        # Rate limiting implementation
        class RateLimiter:
            def __init__(self, requests_per_minute):
                self.requests_per_minute = requests_per_minute
                self.interval = 60.0 / requests_per_minute  # seconds between requests
                self.last_request_time = 0
                
            async def wait(self):
                current_time = time.time()
                elapsed = current_time - self.last_request_time
                
                if elapsed < self.interval:
                    # Wait the remaining time
                    wait_time = self.interval - elapsed
                    print(f"Rate limiting: waiting {wait_time:.2f} seconds before next request...")
                    await asyncio.sleep(wait_time)
                
                self.last_request_time = time.time()
        
        # Create rate limiter - default to 60 RPM but use command line arg if provided
        rate_limiter = RateLimiter(args.requests_per_minute)
        
        # Replace the model_id in the API call
        async def wrapped_process_batch(batch_items, batch_num, total_batches):
            print(f"Running batch {batch_num}/{total_batches} with {len(batch_items)} items using model {args.model_name}")
            print(f"Rate limiting: {args.requests_per_minute} requests per minute")
            batch_start_time = time.time()
            
            results = []
            for item in batch_items:
                try:
                    jailbreak_name = item["jailbreak_name"]
                    jailbreak_fn = item["jailbreak_fn"]
                    prompt = item["prompt"]
                    
                    # Apply the jailbreak function to the prompt
                    jailbroken_content = jailbreak_fn(prompt)
                    
                    # Create a Prompt object with ChatMessage
                    jailbroken_prompt = Prompt(messages=[ChatMessage(content=jailbroken_content, role=MessageRole.user)])
                    
                    print(f"Sending request for: {jailbreak_name} - {prompt[:30]}...")
                    
                    # Apply rate limiting before making the API call
                    await rate_limiter.wait()
                    
                    # Make the actual API call to the model with the specified model_name
                    response = await api(
                        model_id=args.model_name,
                        prompt=jailbroken_prompt,
                        max_attempts_per_api_call=1,
                        force_provider="openrouter",  # Use OpenRouter for wider model support
                    )
                    
                    # Extract the completion from the response (handle different output formats)
                    try:
                        if hasattr(response, 'completions') and response.completions:
                            completion = response.completions[0].text
                        elif hasattr(response, 'completion'):
                            completion = response.completion
                        elif hasattr(response[0], 'completion'):
                            completion = response[0].completion
                        else:
                            completion = str(response)
                    except Exception as extract_error:
                        print(f"Error extracting completion: {extract_error}")
                        completion = "Error extracting response"
                    
                    # Add timestamp to the results
                    current_time = int(time.time())
                    results.append({
                        "prompt": prompt,
                        "jailbreak": jailbreak_name,
                        "jailbroken_prompt": jailbroken_content,  # Store only the content string
                        "response": completion,
                        "timestamp": current_time,
                        "model": args.model_name  # Include model name in results
                    })
                    
                except Exception as e:
                    print(f"Error processing item: {e}")
                    current_time = int(time.time())
                    results.append({
                        "prompt": item["prompt"],
                        "jailbreak": item["jailbreak_name"],
                        "jailbroken_prompt": f"ERROR: {str(e)}",
                        "response": f"ERROR: {str(e)}",
                        "timestamp": current_time,
                        "model": args.model_name  # Include model name in results
                    })
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            print(f"Batch {batch_num} completed in {batch_duration:.2f} seconds")
            
            return results
        
        return await wrapped_process_batch(batch_items, batch_num, total_batches)
    
    # Replace the original process_batch with our wrapped version
    process_batch = process_batch_with_model
    
    return await run_jailbreak_prompt_cartesian_product(
        max_prompts=args.max_prompts, 
        max_jailbreaks=args.max_jailbreaks,
        model_name=args.model_name,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    # Run the main function
    try:
        loop = asyncio.get_event_loop()
        jailbreak_results_df = loop.run_until_complete(main())
        print("Results processing complete")
    except Exception as e:
        print(f"Error in main function: {e}")