import json
import re
import pandas as pd
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import solver, TaskState, Generate, basic_agent, system_message
from inspect_ai.scorer import scorer, Score, Target, accuracy, stderr
from inspect_ai.tool import tool, bash, python

# --- 1. Constants & Prompts ---

SYSTEM_MESSAGE = """You are an expert code auditor evaluating agentic benchmarks using the Agentic Benchmark Checklist (ABC).

## Your Role
You are systematically auditing agentic benchmarks. Your job is to assess whether a benchmark meets specific quality criteria.

## Checklist Task
You will be given a specific question from the ABC framework (e.g., "Is the evaluation code isolated?").
You must investigate the repository and paper to determine if the answer is **YES** or **NO**.

## Resources
- **Repository:** `/benchmark_repo`
- **Paper:** `/paper_latex` (if available)

## Instructions
1. **Explore:** Use `bash` and `python` to search the codebase and paper for evidence.
2. **Reason:** Explain your findings clearly in the chat.
3. **Answer:** When you have reached a conclusion, call the `submit` tool with your final answer.
   - If the criteria is met: `submit(answer="YES")`
   - If the criteria is NOT met: `submit(answer="NO")`

## Rules
- You must provide evidence for your decision in the text before submitting.
- You must strictly use "YES" or "NO" as the argument for the submit tool.
"""

# --- 2. Data Loading ---

def load_benchmark_samples(benchmark_name: str):
    """Load checklist samples for a specific benchmark."""
    base_dir = Path(__file__).parent
    targets_path = base_dir / "data" / "targets.json"
    checklist_path = base_dir / "data" / "checklist.csv"
    
    with open(targets_path, "r") as f:
        all_targets = json.load(f)
    
    if benchmark_name not in all_targets:
        raise ValueError(f"Benchmark '{benchmark_name}' not found in targets.json")
    
    bench_data = all_targets[benchmark_name]
    repo_url = bench_data.get("_repo_url")
    paper_url = bench_data.get("_paper_url", "")
    
    df = pd.read_csv(checklist_path)
    
    samples = []
    for _, row in df.iterrows():
        q_id = row['id']
        target_val = bench_data.get(q_id)
        if target_val is None:
            continue
        
        target_str = "YES" if str(target_val) in ["1", "1.0", "YES", "True"] else "NO"
        
        samples.append(
            Sample(
                input=row['question'],
                target=target_str,
                metadata={
                    "id": q_id,
                    "category": row['category'],
                    "subcategory": row['subcategory'],
                    "benchmark_name": benchmark_name,
                    "repo_url": repo_url,
                    "paper_url": paper_url,
                }
            )
        )
    return samples

# --- 3. Tools ---

@tool
def submit():
    """
    Submit the final answer to the evaluation.
    """
    async def execute(answer: str):
        """
        Record the final answer and end the task.

        Args:
            answer: The final decision, must be 'YES' or 'NO'.
        """
        return f"Answer '{answer}' submitted successfully."

    return execute

# --- 4. Solvers ---

@solver
def setup_environment():
    """Clones repo and downloads paper."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        repo_url = state.metadata["repo_url"]
        paper_url = state.metadata.get("paper_url", "")
        
        # Handle tree URLs
        if "/tree/" in repo_url:
            match = re.match(r"(https://github\.com/[^/]+/[^/]+)", repo_url)
            base_repo_url = match.group(1) if match else repo_url
        else:
            base_repo_url = repo_url

        # Clone
        await bash(timeout=180)(f"git clone {base_repo_url} /benchmark_repo")
        
        # Download Paper
        if paper_url:
            arxiv_match = re.search(r"arxiv\.org/pdf/(\d+\.\d+)", paper_url)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                await bash(timeout=120)(
                    f"mkdir -p /paper_latex && "
                    f"curl -L https://arxiv.org/src/{arxiv_id} -o /tmp/paper.tar.gz && "
                    f"tar -xzf /tmp/paper.tar.gz -C /paper_latex"
                )
        return state
    return solve

@solver
def format_challenge():
    """Injects specific question details into the prompt."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        q_id = state.metadata.get('id', '')
        prompt = (
            f"--- CHECKLIST ITEM {q_id} ---\n"
            f"Category: {state.metadata.get('category')}\n"
            f"Question: {state.input_text}\n\n"
            f"Remember to investigate using tools, then call submit(answer='YES') or submit(answer='NO')."
        )
        state.user_prompt.text = prompt
        return state
    return solve

# --- 5. Scorer ---

@scorer(metrics=[accuracy(), stderr()])
def submission_scorer():
    """
    Scans the conversation history for a structured tool call to 'submit'.
    """
    async def score(state: TaskState, target: Target) -> Score:
        predicted = None
        
        # Scan messages in reverse order to find the latest tool call to 'submit'
        for message in reversed(state.messages):
            # Only Assistant messages contain tool_calls. 
            # We must check the role or use getattr to avoid AttributeError on Tool/User messages.
            if message.role == "assistant" and getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    if tool_call.function == "submit":
                        # Inspect parses the arguments for us
                        predicted = tool_call.arguments.get("answer")
                        break
            if predicted:
                break
        
        if not predicted:
            return Score(
                value=0.0,
                answer="N/A",
                explanation="Agent did not call the submit() tool."
            )
            
        # Normalize inputs (handles mixed casing)
        predicted_clean = str(predicted).strip().upper()
        truth = target.text.strip().upper()
        
        is_correct = (predicted_clean == truth)
        
        return Score(
            value=1.0 if is_correct else 0.0,
            answer=predicted_clean,
            explanation=f"Agent submitted: {predicted_clean}. Ground truth: {truth}."
        )
    return score

# --- 6. Main Task ---

@task
def benchmark_eval(benchmark_name: str):
    samples = load_benchmark_samples(benchmark_name)
    
    return Task(
        dataset=MemoryDataset(samples),
        plan=[
            setup_environment(),
            format_challenge(),
            basic_agent(
                init=system_message(SYSTEM_MESSAGE),
                tools=[submit(), bash(), python()],
                max_messages=200,
            ),
        ],
        scorer=submission_scorer(),
        sandbox="docker",
        time_limit=600,
    )