import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import io
import altair as alt
import time
from typing import Dict, List, Tuple

def create_model_variables(model: cp_model.CpModel, 
                         tasks: List[Dict], 
                         machines: List[int], 
                         machine_columns: List[str], 
                         horizon: int) -> Tuple[Dict, List[List[int]]]:
    """Create and return all model variables."""
    times = [[t[col] for col in machine_columns] for t in tasks]
    
    variables = {
        'start': {},
        'intervals': {},
        'task_end': {},
    }
    
    for task_idx, _ in enumerate(tasks):
        for machine_idx in machines:
            duration = times[task_idx][machine_idx]
            
            start_var = model.NewIntVar(0, horizon, f'start_{task_idx}_m{machine_idx}')
            interval_var = model.NewIntervalVar(
                start_var, duration, start_var + duration, 
                f'interval_{task_idx}_m{machine_idx}'
            )
            
            variables['start'][(task_idx, machine_idx)] = start_var
            variables['intervals'][(task_idx, machine_idx)] = interval_var
        
        variables['task_end'][task_idx] = model.NewIntVar(
            0, horizon, f'end_time_task{task_idx}'
        )
    
    return variables, times

def add_scheduling_constraints(model: cp_model.CpModel, 
                             tasks: List[Dict], 
                             machines: List[int], 
                             variables: Dict, 
                             times: List[List[int]]) -> None:
    """Add all scheduling constraints to the model."""
    for machine_idx in machines:
        machine_intervals = [
            variables['intervals'][(task_idx, machine_idx)] 
            for task_idx, _ in enumerate(tasks)
        ]
        model.AddNoOverlap(machine_intervals)
    
    for task_idx, _ in enumerate(tasks):
        task_intervals = [
            variables['intervals'][(task_idx, machine_idx)] 
            for machine_idx in machines
        ]
        model.AddNoOverlap(task_intervals)
        
        for machine_idx in machines:
            model.Add(variables['start'][(task_idx, machine_idx)] >= 
                     tasks[task_idx]['ReleaseDate'])
        
        end_times_task = [
            variables['start'][(task_idx, machine_idx)] + times[task_idx][machine_idx]
            for machine_idx in machines
        ]
        model.AddMaxEquality(variables['task_end'][task_idx], end_times_task)

def create_objective_variables(model: cp_model.CpModel, 
                             tasks: List[Dict], 
                             variables: Dict, 
                             horizon: int) -> List:
    """Create and return tardiness variables for the objective function."""
    return [
        model.NewIntVar(0, horizon, f'lateness_task{idx}') * task['Weight']
        for idx, task in enumerate(tasks)
        if model.Add(_ >= variables['task_end'][idx] - task['DueDate']) 
        and model.Add(_ >= 0)
    ]

def solve_scheduling_problem(df: pd.DataFrame, 
                           machine_columns: List[str]) -> Dict:
    """Solve the scheduling problem and return results."""
    tasks = df.to_dict('records')
    machines = list(range(len(machine_columns)))
    
    horizon = max(
        max(t['DueDate'] for t in tasks),
        sum(max(t[col] for col in machine_columns) for t in tasks)
    )
    
    model = cp_model.CpModel()
    variables, times = create_model_variables(
        model, tasks, machines, machine_columns, horizon
    )
    
    add_scheduling_constraints(model, tasks, machines, variables, times)
    
    tardiness_vars = create_objective_variables(model, tasks, variables, horizon)
    model.Minimize(sum(tardiness_vars))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    
    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time
    
    results = {
        'status': status, 
        'objective': None, 
        'schedule': [], 
        'solve_time': solve_time
    }
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        results['objective'] = solver.ObjectiveValue()
        results['schedule'] = extract_solution(solver, tasks, machines, variables, times)
    
    return results

def validate_schedule(schedule: List[Dict], 
                     input_data: pd.DataFrame, 
                     machine_columns: List[str],
                     status: int) -> Dict:
    """Validate the schedule against constraints."""
    task_ids = set(input_data["TaskID"].tolist())
    schedule_task_ids = {task["task_id"] for task in schedule}
    
    results = {}
    
    # Check all tasks are handled
    missing_tasks = task_ids - schedule_task_ids
    results["all_tasks_handled"] = (
        not missing_tasks,
        f"Missing tasks: {list(missing_tasks)}" if missing_tasks else 
        "All tasks are handled."
    )
    
    # Check release dates
    early_starts = []
    for task in schedule:
        task_id = task["task_id"]
        release_date = input_data.loc[
            input_data["TaskID"] == task_id, "ReleaseDate"
        ].iloc[0]
        
        early_starts.extend(
            f"Task {task_id} on Machine {m_num} starts early: {start} < {release_date}"
            for m_num, start, _ in task["machine_times"]
            if start < release_date
        )
    
    results["no_early_start"] = (
        not early_starts,
        "\n".join(early_starts) if early_starts else "No early starts."
    )
    
    # Additional validation checks would follow similar pattern...
    
    results["Optimal solution"] = (
        status == cp_model.OPTIMAL,
        "Optimal solution found" if status == cp_model.OPTIMAL 
        else "Feasible but not optimal solution"
    )
    
    return results

def main() -> None:
    """Main application function with reduced complexity."""
    st.set_page_config(
        page_title="Multi-Machine Scheduling Optimizer",
        page_icon="🛠️",
        layout="wide"
    )
    
    setup_sidebar()
    setup_main_page()
    
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
    if not uploaded_file:
        st.info("Upload an Excel file to start.")
        return
        
    df = load_and_validate_data(uploaded_file)
    if df is None:
        return
        
    machine_columns = setup_machine_columns(df)
    
    if st.button("Solve Scheduling Problem"):
        process_scheduling_solution(df, machine_columns)

# Additional helper functions to break down main's complexity
def setup_sidebar() -> None:
    """Setup the sidebar with instructions."""
    with st.sidebar:
        st.title("Upload task data")
        st.markdown(
            """
            1. Upload an Excel file (.xlsx) with:
               - **TaskID** (unique identifier)
               - **ReleaseDate**
               - **DueDate**
               - **Weight**
               - **Machine1Time**, **Machine2Time**, etc.
            2. Configure detected machine columns below.
            3. Click **Solve Scheduling Problem** to optimize.
            """
        )
        st.info("Ensure correct file format to avoid errors.")

def setup_main_page() -> None:
    """Setup the main page with title and description."""
    st.title("Multi-Machine Scheduling Optimizer")
    st.markdown(
        """
        Optimize multi-machine scheduling tasks to minimize total 
        **weighted tardiness**.  
        Use the **sidebar** to upload data and configure settings.
        """
    )

def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate the uploaded data."""
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.map(str)
        
        if not validate_columns(df):
            return None
            
        if df.isnull().values.any():
            display_empty_cells(df)
            return None
            
        st.markdown("### Input Data Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
