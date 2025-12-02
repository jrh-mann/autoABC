import json
import os
import re
from pathlib import Path
import pandas as pd
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import match
from inspect_ai.solver import (
    basic_agent, 
    system_message, 
    solver, 
    TaskState
)
from inspect_ai.tool import bash, python

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# --- 1. Environment Setup Tools ---

def extract_repo_info(github_url: str):
    """
    Extracts base repo URL and subdirectory path from GitHub URLs.
    Handles URLs like:
    - https://github.com/user/repo (no subdirectory)
    - https://github.com/user/repo/tree/main/path (with subdirectory)
    """
    # Remove /tree/... or /blob/... parts to get base repo URL
    base_url = re.sub(r'/tree/[^/]+(/.*)?$', '', github_url)
    base_url = re.sub(r'/blob/[^/]+(/.*)?$', '', base_url)
    
    # Extract subdirectory if present
    subdir_match = re.search(r'/tree/[^/]+(/.*)$', github_url)
    subdir = subdir_match.group(1) if subdir_match else None
    
    return base_url, subdir

@solver
def setup_audit_environment(repo_url: str, paper_url: str, repo_subdir: str = None):
    """
    Sets up the sandbox with both the Code (repo) and the Paper (latex source).
    
    Args:
        repo_url: Base GitHub repository URL (without /tree/... paths)
        paper_url: URL to the paper (arXiv PDF or other)
        repo_subdir: Optional subdirectory path within the repo (e.g., "project/swelancer")
    """
    async def solve(state: TaskState, generate):
        # A. Clone the Target Repo
        # Check if already cloned to handle persistent environments
        clone_cmd = f"if [ ! -d '/audit_target' ]; then git clone {repo_url} /audit_target; fi"
        await state.tools["bash"].execute(clone_cmd)
        
        # If there's a subdirectory, note it for the agent
        if repo_subdir:
            # Remove leading slash if present
            repo_subdir = repo_subdir.lstrip('/')
            await state.tools["bash"].execute(
                f"echo 'Benchmark code is in /audit_target/{repo_subdir}' > /audit_target/.benchmark_location"
            )
        
        # B. Download the Paper (LaTeX Source)
        # Handle arXiv URLs: Convert 'arxiv.org/abs/XXXX.XXXXX' to 'arxiv.org/e-print/XXXX.XXXXX'
        if "arxiv.org" in paper_url:
            # Extract arXiv ID (format: YYMM.NNNNN or YYMM.NNNNNvN)
            arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', paper_url)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                src_url = f"https://arxiv.org/e-print/{arxiv_id}"
            else:
                # Fallback: try simple replacement
                src_url = paper_url.replace("/abs/", "/e-print/").replace("/pdf/", "/e-print/").replace(".pdf", "")
            
            setup_cmds = [
                "mkdir -p /audit_paper",
                # Download with -L to follow redirects (essential for arXiv)
                f"curl -L {src_url} -o /audit_paper/source.tar.gz", 
                # Extract the tarball (handle both .tar.gz and .gz)
                "cd /audit_paper && tar -xzf source.tar.gz 2>/dev/null || gunzip -c source.tar.gz | tar -xf -",
                # Cleanup
                "rm -f /audit_paper/source.tar.gz"
            ]
            
            # Execute as a single chained command for robustness
            await state.tools["bash"].execute(" && ".join(setup_cmds))
        
        else:
            # Fallback for non-arXiv links (simple wget)
            await state.tools["bash"].execute(
                f"mkdir -p /audit_paper && wget {paper_url} -P /audit_paper 2>/dev/null || curl -L {paper_url} -o /audit_paper/paper.pdf"
            )

        # C. Orient the Agent
        location_hint = ""
        if repo_subdir:
            location_hint = f"\n- Note: The benchmark code may be in a subdirectory `/audit_target/{repo_subdir.lstrip('/')}`"
        
        state.messages.append(system_message(
            f"ENVIRONMENT SETUP COMPLETE.\n"
            f"- Codebase: Located in `/audit_target` (Use this for T.* checks){location_hint}\n"
            f"- Paper Source: Located in `/audit_paper` (Use this for R.* and O.* checks)\n"
            f"You can read the .tex files in /audit_paper to find sections about methodology and reporting."
        ))
        
        return state
    return solve

# --- 2. Dynamic Task Factory ---

def get_available_benchmarks():
    """Returns list of available benchmark names from targets.json"""
    targets_path = SCRIPT_DIR / "data" / "targets.json"
    if not targets_path.exists():
        return []
    with open(targets_path, "r") as f:
        all_targets = json.load(f)
    return list(all_targets.keys())

def create_dataset_for_benchmark(benchmark_name: str):
    # Load questions (use absolute path relative to script location)
    checklist_path = SCRIPT_DIR / "data" / "checklist.csv"
    if not checklist_path.exists():
        raise FileNotFoundError(f"Checklist file not found: {checklist_path}")
    df_questions = pd.read_csv(checklist_path)
    
    # Load targets (The "Free Ride" data)
    targets_path = SCRIPT_DIR / "data" / "targets.json"
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets file not found: {targets_path}")
    with open(targets_path, "r") as f:
        all_targets = json.load(f)
        
    benchmark_data = all_targets.get(benchmark_name)
    if not benchmark_data:
        raise ValueError(f"No ground truth found for {benchmark_name}")

    samples = []
    for _, row in df_questions.iterrows():
        q_id = row['id']
        question = row['question']
        category = row['category']
        
        # Determine strictness of target (convert 0/1 to YES/NO)
        target_val = benchmark_data.get(q_id, "N/A")
        target_str = "YES" if target_val == 1 else "NO"
        
        # Add a hint about where to look based on the Category
        # T.* = Task Validity -> Code
        # R.* = Reporting -> Paper
        # O.* = Outcome -> Both
        hint = "Check the code in /audit_target."
        if category == "Benchmark Reporting":
            hint = "Search the LaTeX files in /audit_paper."
        elif category == "Outcome Validity":
            hint = "Check both the paper (for claims) and the code (for implementation)."

        samples.append(Sample(
            id=q_id,
            input=f"QUESTION ({q_id}): {question}\n\nHint: {hint}\nAnswer only with YES or NO.",
            target=target_str,
            metadata={
                "category": category,
                "benchmark": benchmark_name
            }
        ))
    
    # Extract repo URL and subdirectory
    repo_url_full = benchmark_data.get("_repo_url", "")
    repo_url_base, repo_subdir = extract_repo_info(repo_url_full)
    paper_url = benchmark_data.get("_paper_url", "")
    
    return MemoryDataset(samples), repo_url_base, paper_url, repo_subdir

@task
def agentic_audit(benchmark_name: str = None):
    """
    Audits an agentic benchmark against the ABC (Agentic Benchmark Checklist).
    
    Args:
        benchmark_name: Name of the benchmark to audit (must exist in targets.json).
                       If not provided, reads from BENCHMARK_NAME environment variable.
    
    Examples:
        # Using environment variable:
        BENCHMARK_NAME="SWE-Lancer" inspect eval autoABC.py::agentic_audit
        
        # Or set it in your shell:
        export BENCHMARK_NAME="SWE-Lancer"
        inspect eval autoABC.py::agentic_audit
    """
    # Get benchmark name from parameter or environment variable
    if benchmark_name is None:
        benchmark_name = os.getenv("BENCHMARK_NAME")
        if not benchmark_name:
            raise ValueError(
                "benchmark_name must be provided either as a parameter or via "
                "BENCHMARK_NAME environment variable. Available benchmarks: "
                + ", ".join(get_available_benchmarks())
            )
    
    # 1. Get Data
    dataset, repo_url, paper_url, repo_subdir = create_dataset_for_benchmark(benchmark_name)
    
    # 2. Define the Task
    return Task(
        dataset=dataset,
        plan=[
            # Setup: Clones Repo + Downloads Paper
            setup_audit_environment(repo_url, paper_url, repo_subdir),
            
            # Agent: Standard tools (bash/python) are sufficient for reading files
            system_message("You are an expert AI Benchmark Auditor. You verify if code meets specific scientific standards."),
            basic_agent(tools=[bash(timeout=60), python(timeout=60)])
        ],
        scorer=match(),
        sandbox="docker",
    )

# --- 3. Individual Task Functions for Each Benchmark ---
# These can be called directly without environment variables

@task
def audit_swe_lancer():
    """Audit SWE-Lancer benchmark"""
    return agentic_audit("SWE-Lancer")

@task
def audit_bird_bench():
    """Audit Bird-Bench benchmark"""
    return agentic_audit("Bird-Bench")

@task
def audit_cybench():
    """Audit CyBench benchmark"""
    return agentic_audit("CyBench")

@task
def audit_swe_bench_verified():
    """Audit SWE-bench-Verified benchmark"""
    return agentic_audit("SWE-bench-Verified")

@task
def audit_tau_bench():
    """Audit tau-Bench benchmark"""
    return agentic_audit("tau-Bench")

@task
def audit_mle_bench():
    """Audit MLE-Bench benchmark"""
    return agentic_audit("MLE-Bench")

@task
def audit_webarena():
    """Audit WebArena benchmark"""
    return agentic_audit("WebArena")

@task
def audit_osworld():
    """Audit OSWorld benchmark"""
    return agentic_audit("OSWorld")

@task
def audit_kernelbench():
    """Audit KernelBench benchmark"""
    return agentic_audit("KernelBench")