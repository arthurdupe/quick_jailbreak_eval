#%%

from IPython import get_ipython
ipython = get_ipython()

if ipython:
    # If running in IPython, use nest_asyncio to allow nested event loops
    print("IPython detected, applying nest_asyncio patch...")
    import nest_asyncio
    nest_asyncio.apply()
    print("nest_asyncio patch applied successfully")
        
    # Run the magic commands
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
    print("IPython magic commands executed: autoreload enabled")
else:
    # Not running in IPython, skip magic commands
    print("Not running in IPython environment, skipping magic commands")

import asyncio
import os
from pathlib import Path
import time # Import time for basic timing, though api.log_model_timings is better

# Import necessary components from the safety-tooling library
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from prompt_28_apr import jailbreaks

# --- Configuration ---
# You can choose any model available on OpenRouter
# Check https://openrouter.ai/docs for model IDs
OPENROUTER_MODEL_ID = "google/gemini-flash-1.5-8b" # An inexpensive model for testing
# How many requests to send in parallel to OpenRouter
# Adjust based on your OpenRouter rate limits and local resources
OPENROUTER_CONCURRENCY = 10
# Set this to False to disable caching for testing parallel calls, True otherwise
USE_CACHE = False
# Set to True to print prompts and responses directly from the API handler
PRINT_API_IO = False
# Set to True to run basic arithmetic tests (1+1, 2+2, etc.)
TEST_BASIC = False

# --- Environment Setup ---
# Loads API keys from the .env file and sets up basic logging
# Make sure you have a .env file with OPENROUTER_API_KEY set.
utils.setup_environment()

# Check if OPENROUTER_API_KEY is set before proceeding
if "OPENROUTER_API_KEY" not in os.environ or not os.environ["OPENROUTER_API_KEY"]:
    print("Error: OPENROUTER_API_KEY environment variable is not set.")
    print("Please set it in your .env file before running this notebook.")
else:
    print("OPENROUTER_API_KEY found. Proceeding with API setup.")

# --- Temporary fix: Add dummy OpenAI API key ---
# The InferenceAPI tries to initialize OpenAI clients even when we only want to use OpenRouter
# This is a workaround to prevent the initialization error
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("Setting dummy OPENAI_API_KEY to bypass initialization error")
    os.environ["OPENAI_API_KEY"] = "dummy-key-not-used"

#%% [markdown]
# ## Initialize the Inference API
#
# Here we create the `InferenceAPI` instance. We configure it to use OpenRouter and set the desired concurrency limit. Caching is disabled for this example to ensure API calls are made in parallel.

#%%
# --- Inference API Initialization ---
# Instantiate the InferenceAPI.
# The 'openrouter_num_threads' parameter controls the maximum concurrency
# for calls specifically to the OpenRouter provider.
# We disable caching here for demonstration purposes to ensure calls hit the API.

api = InferenceAPI(
    openrouter_num_threads=OPENROUTER_CONCURRENCY,
    cache_dir=Path(".cache") if USE_CACHE else None, # Set cache_dir=None to disable caching
    prompt_history_dir=Path("prompt_history"), # Directory to save prompt history (optional)
    print_prompt_and_response=PRINT_API_IO, # Whether to print IO directly from API handler
)

print(f"InferenceAPI initialized for OpenRouter model {OPENROUTER_MODEL_ID}")
print(f"Configured concurrency for OpenRouter: {api.openrouter_num_threads}")
print(f"Caching enabled: {api.cache_manager is not None}")
print(f"Prompt history enabled: {api.prompt_history_dir is not None}")


#%% [markdown]
# ## Define Prompts
#
# We create a list of simple prompts to send to the LLM.

#%%
# --- Define Prompts ---
# Create a list of Prompt objects for parallel execution
if TEST_BASIC:
    prompts_to_run = [
        Prompt(messages=[ChatMessage(content="What is 1+1?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 2+2?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 3+3?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 4+4?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 5+5?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 6+6?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 7+7?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 8+8?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 9+9?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 10+10?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 11+11?", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="What is 12+12?", role=MessageRole.user)]),
    ]
    
    print(f"Defined {len(prompts_to_run)} prompts to run.")
else:
    print("Skipping basic arithmetic tests (TEST_BASIC is False)")
    prompts_to_run = []

#%% [markdown]
# ## Run Parallel Inference
#
# This cell uses `asyncio.gather` to execute the API calls for each prompt concurrently. The `InferenceAPI` object's internal logic will manage the concurrency according to the `openrouter_num_threads` setting.

#%%
# --- Run Parallel Inference ---
if TEST_BASIC and prompts_to_run:
    print(f"Starting parallel inference for {len(prompts_to_run)} prompts using {OPENROUTER_MODEL_ID}")
    
    async def run_basic_tests():
        start_time = time.time()
        
        # Create a list of asynchronous tasks
        tasks = []
        for i, prompt in enumerate(prompts_to_run):
            # Each call to the API returns a list of responses (usually 1 if n=1)
            # Pass force_provider="openrouter" if the model_id isn't automatically recognized by safety-tooling
            # as an OpenRouter model (check the OPENROUTER_MODELS set in api.py and openrouter.py)
            task = api(
                model_id=OPENROUTER_MODEL_ID,
                prompt=prompt,
                n=1, # Request 1 completion per prompt
                max_tokens=50, # Limit response length
                temperature=0.0, # For deterministic answers (often ignored by models)
                force_provider="openrouter", # Force using OpenRouter to be safe
                print_prompt_and_response=False, # We'll print manually below
            )
            tasks.append(task)
        
        # Run tasks concurrently and wait for all to complete
        # asyncio.gather ensures these tasks run in parallel within the event loop
        # The 'api' object's internal semaphore respects the 'openrouter_num_threads' limit
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("\n--- Inference Complete ---")
        print(f"Total time taken for {len(prompts_to_run)} prompts: {total_duration:.2f} seconds")
        print(f"Average time per prompt: {total_duration / len(prompts_to_run):.2f} seconds")
        
        return results
    
    # Run the async function in a way that works in both notebook and script environments
    if ipython:
        # In Jupyter/IPython, we can use the nest_asyncio patched event loop
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(run_basic_tests())
    else:
        # In a standalone script, use asyncio.run
        results = asyncio.run(run_basic_tests())
else:
    print("Skipping parallel inference (TEST_BASIC is False or no prompts defined)")
    results = []


#%% [markdown]
# ## Display Results
#
# Iterate through the results and print each prompt and its corresponding response.

#%%
# --- Display Results ---
if TEST_BASIC and prompts_to_run and results:
    print("\n--- Results ---")
    for i, (prompt, responses) in enumerate(zip(prompts_to_run, results)):
        print(f"Prompt {i+1}: {prompt.messages[0].content}")
        if responses and len(responses) > 0 and responses[0]:
            # Assuming n=1, responses is a list with one LLMResponse object
            response_text = responses[0].completion.strip() if responses[0].completion else "Empty completion"
            stop_reason = responses[0].stop_reason
            print(f"Response {i+1}: {response_text}")
            print(f"Stop Reason: {stop_reason}")
        else:
             print(f"Response {i+1}: No response or error received (check logs for errors)")
        print("-" * 20)
else:
    print("Skipping results display (TEST_BASIC is False, no prompts defined, or no results available)")

#%% [markdown]
# ## Optional: Log Timings and Cost
#
# If `prompt_history_dir` was set during API initialization, you can log detailed model timings. The API also tracks estimated cost.

#%%
# Optional: Log model timings if prompt_history_dir is set
if TEST_BASIC and api.prompt_history_dir:
     print("\n--- Model Timings ---")
     # This will print timing info to the console and save a plot if matplotlib is available
     api.log_model_timings()

# Optional: Log total cost if any models with costs were used
# Note: Cost calculation for OpenRouter might not be perfectly accurate within the library,
# but it provides an estimate based on OpenAI pricing for similar models where possible,
# or defaults to 0 if no pricing info is available.
if TEST_BASIC and api.running_cost > 0:
    print(f"\nTotal estimated cost: ${api.running_cost:.6f}")
elif TEST_BASIC:
     print("\nTotal estimated cost: $0 (Cost tracking might not be available for this model or provider, or no tokens were generated)")

# %%

#%% [markdown]
# ## Reading Parquet Data
#
# This cell reads data from a parquet file for evaluation.

#%%
# --- Read Parquet Data ---
import pandas as pd

data_path = Path("/workspace/jailbreaking/data/train-00000-of-00001.parquet")
if data_path.exists():
    df = pd.read_parquet(data_path)
    print(f"Loaded parquet data: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Display sample row
    if len(df) > 0:
        print("\nSample data:")
        sample = df.iloc[0]
        for col in df.columns:
            value = sample[col]
            print(f"{col}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
else:
    print(f"Error: Parquet file not found at {data_path}")

# Function to create prompts from dataframe rows
def create_prompts_from_df(df, num_samples=5):
    # Sample a subset of the dataframe
    if num_samples is None:
        # Use the entire dataframe if num_samples is None
        sample_df = df
    elif len(df) > num_samples:
        sample_df = df.sample(n=num_samples, random_state=42)
    else:
        sample_df = df
    
    prompts = []
    
    # Assuming the dataframe has columns like 'prompt' or 'question'
    # Adjust the column names based on your actual data structure
    text_column = next((col for col in ['prompt', 'question', 'input', 'text'] 
                       if col in df.columns), df.columns[0])
    
    for _, row in sample_df.iterrows():
        content = row[text_column]
        prompt = Prompt(messages=[ChatMessage(content=content, role=MessageRole.user)])
        prompts.append(prompt)
    
    return prompts

# Sample usage of creating prompts from parquet data
if 'df' in locals() and len(df) > 0:
    print("\n--- Creating Prompts from Parquet Data ---")
    parquet_prompts = create_prompts_from_df(df, num_samples=3)
    print(f"Created {len(parquet_prompts)} prompts from parquet data")
    for i, prompt in enumerate(parquet_prompts):
        print(f"Parquet Prompt {i+1}: {prompt.messages[0].content[:100]}...")

# %%

#%% [markdown]
# ## Run Cartesian Product of Jailbreaks and Prompts
#
# This cell runs all combinations of jailbreaks and prompts from the parquet data, and saves the results to a dataframe.

#%%
# --- Run Cartesian Product of Jailbreaks and Prompts ---
import itertools
from datetime import datetime

async def run_jailbreak_prompt_cartesian_product(max_prompts=3, max_jailbreaks=None, dataframe=None):
    """
    Run a Cartesian product of jailbreaks and prompts, and save results to a dataframe.
    
    Args:
        max_prompts: Maximum number of prompts to use from parquet data
        max_jailbreaks: Maximum number of jailbreaks to use (None = all)
        dataframe: DataFrame to use, if None will try to use global df
    """
    # Get prompts from parquet data
    global df
    
    # Use provided dataframe or fallback to global df
    if dataframe is not None:
        data = dataframe
    elif 'df' in globals() and globals()['df'] is not None and len(globals()['df']) > 0:
        data = df
    else:
        print("No parquet data available.")
        return None
    
    sample_prompts = create_prompts_from_df(data, num_samples=max_prompts)
    prompt_contents = [p.messages[0].content for p in sample_prompts]
    print(f"Using {len(prompt_contents)} prompts from parquet data")
    
    # Get jailbreaks
    all_jailbreaks = list(jailbreaks.JAILBREAKS.items())
    if max_jailbreaks is not None:
        all_jailbreaks = all_jailbreaks[:max_jailbreaks]
    print(f"Using {len(all_jailbreaks)} jailbreak methods")
    
    # Create all combinations
    print(f"Creating {len(prompt_contents) * len(all_jailbreaks)} total combinations")
    combinations = list(itertools.product(all_jailbreaks, prompt_contents))
    
    # Create list to store results
    results_data = []
    jailbreak_prompts = []
    
    # Prepare jailbroken prompts
    for (jailbreak_name, jailbreak_func), prompt_content in combinations:
        # Apply the jailbreak function to the prompt
        jailbroken_content = jailbreak_func(prompt_content)
        # Create a new prompt with the jailbroken content
        jailbroken_prompt = Prompt(messages=[ChatMessage(content=jailbroken_content, role=MessageRole.user)])
        jailbreak_prompts.append({
            'jailbreak_name': jailbreak_name,
            'original_prompt': prompt_content,
            'jailbroken_prompt': jailbroken_prompt
        })
    
    # Run in batches to manage concurrency
    batch_size = OPENROUTER_CONCURRENCY
    num_batches = (len(jailbreak_prompts) + batch_size - 1) // batch_size
    print(f"Running {len(jailbreak_prompts)} API calls in {num_batches} batches of size {batch_size}")
    
    overall_start_time = time.time()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(jailbreak_prompts))
        
        batch_items = jailbreak_prompts[start_idx:end_idx]
        print(f"Running batch {batch_idx+1}/{num_batches} with {len(batch_items)} items")
        
        batch_start_time = time.time()
        
        # Create tasks for this batch
        tasks = []
        for item in batch_items:
            task = api(
                model_id=OPENROUTER_MODEL_ID,
                prompt=item['jailbroken_prompt'],
                n=1,
                max_tokens=200,
                temperature=0.0,
                force_provider="openrouter",
            )
            tasks.append(task)
        
        # Run tasks concurrently and wait for all to complete
        batch_results = await asyncio.gather(*tasks)
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        print(f"Batch {batch_idx+1} completed in {batch_duration:.2f} seconds")
        
        # Process results and add to results_data
        for item, responses in zip(batch_items, batch_results):
            if responses and len(responses) > 0 and responses[0]:
                response_text = responses[0].completion.strip() if responses[0].completion else "Empty completion"
                stop_reason = responses[0].stop_reason
                request_time = responses[0].request_time if hasattr(responses[0], 'request_time') else None
                
                results_data.append({
                    'jailbreak_name': item['jailbreak_name'],
                    'original_prompt': item['original_prompt'],
                    'jailbroken_prompt': item['jailbroken_prompt'].messages[0].content,
                    'response': response_text,
                    'stop_reason': stop_reason,
                    'request_time': request_time,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                results_data.append({
                    'jailbreak_name': item['jailbreak_name'],
                    'original_prompt': item['original_prompt'],
                    'jailbroken_prompt': item['jailbroken_prompt'].messages[0].content,
                    'response': "No response or error",
                    'stop_reason': None,
                    'request_time': None,
                    'timestamp': datetime.now().isoformat()
                })
    
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    print(f"\nAll batches completed in {overall_duration:.2f} seconds")
    print(f"Average time per call: {overall_duration / len(jailbreak_prompts):.2f} seconds")
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Add timestamp to filenames to prevent overwriting
    timestamp_suffix = f"_{int(time.time())}"
    
    # Make sure the directory exists
    data_dir = Path("/workspace/jailbreaking/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame to parquet
    output_path = data_dir / f"jailbreak_results{timestamp_suffix}.parquet"
    results_df.to_parquet(output_path)
    print(f"Results saved to {output_path}")
    
    # Also save a CSV for easier viewing
    csv_path = data_dir / f"jailbreak_results{timestamp_suffix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results also saved to {csv_path}")
    
    return results_df

# Run the cartesian product with limited samples for testing
print("Running cartesian product of jailbreaks and prompts...")

# Define the async function that will run our jailbreak tests
async def main():
    return await run_jailbreak_prompt_cartesian_product(max_prompts=None, max_jailbreaks=None, dataframe=df)

# Run the async function in a way that works in both notebook and script environments
if ipython:
    # In Jupyter/IPython, we can use the nest_asyncio patched event loop
    loop = asyncio.get_event_loop()
    jailbreak_results_df = loop.run_until_complete(main())
else:
    # In a standalone script, use asyncio.run
    jailbreak_results_df = asyncio.run(main())

#%%

if jailbreak_results_df is not None:
    print("\n--- Sample Results ---")
    # Display a few sample results
    for _, row in jailbreak_results_df.head(3).iterrows():
        print(f"Jailbreak: {row['jailbreak_name']}")
        print(f"\n\n^^Prompt: {row['jailbroken_prompt'][:1000]}...")
        print(f"\n\n^^Response: {row['response'][:1000]}...")
        print("-" * 50)

# %%
