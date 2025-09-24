# Iteration Efficiency Quantification Formula

## Core Formula for Complex Planned Tasks

```
                    Steps Completed     Quality Score     1
Iteration Efficiency = --------------- × ------------- × ---------------
                     Steps Planned      Max Quality    Rework Factor
```

## 🔢 Detailed Component Breakdown

### 1. Steps Completed / Steps Planned (Completion Rate)
```python
def calculate_completion_rate(session, markdown_plan):
    """
    Measures how many planned steps were actually completed
    """
    # Extract planned steps from markdown
    planned_steps = extract_numbered_items(markdown_plan)
    total_planned = len(planned_steps)

    # Detect completion signals in session
    completion_patterns = [
        r'step.{0,3}(\d+).{0,10}(complete|done|finished)',
        r'(\d+).{0,10}(implemented|working|success)',
        r'moving.{0,5}to.{0,5}(step|item).{0,3}(\d+)',
        r'✓|✅|☑️.{0,10}(\d+)',  # Checkbox patterns
    ]

    completed_count = 0
    for step_num in range(1, total_planned + 1):
        if detect_step_completion(session, step_num, completion_patterns):
            completed_count += 1

    completion_rate = completed_count / total_planned
    return completion_rate  # Range: 0.0 to 1.0
```

### 2. Quality Score Component
```python
def calculate_quality_score(session):
    """
    Measures the quality of completed work
    """
    quality_metrics = {
        'tests_passing': 0.25,      # Did tests pass?
        'requirements_met': 0.25,    # Were requirements satisfied?
        'no_critical_errors': 0.20,  # Absence of breaking errors
        'code_runs': 0.15,          # Does the code execute?
        'user_satisfied': 0.15      # Explicit satisfaction signals
    }

    score = 0.0

    # Test execution quality
    if 'test.*pass' in session or 'all.*tests.*passing' in session:
        score += quality_metrics['tests_passing']

    # Requirements verification
    if 'requirement.*met' in session or 'works.*as.*expected' in session:
        score += quality_metrics['requirements_met']

    # Error analysis
    critical_errors = count_patterns(session, ['fatal', 'critical', 'crash'])
    if critical_errors == 0:
        score += quality_metrics['no_critical_errors']

    # Code execution
    if 'successfully.*ran' in session or 'output:.*success' in session:
        score += quality_metrics['code_runs']

    # User satisfaction
    if detect_gratitude(session) or detect_positive_closure(session):
        score += quality_metrics['user_satisfied']

    return score  # Range: 0.0 to 1.0
```

### 3. Rework Factor (Inverse of Rework Rate)
```python
def calculate_rework_factor(session, steps):
    """
    Penalizes excessive iteration on same steps
    """
    rework_events = []

    for step_num in range(1, len(steps) + 1):
        attempts = count_step_attempts(session, step_num)

        if attempts > 1:
            rework_events.append(attempts - 1)  # Extra attempts

    if not rework_events:
        return 1.0  # Perfect - no rework

    # Calculate rework penalty
    total_rework = sum(rework_events)
    average_rework = total_rework / len(steps)

    # Rework factor (inverse relationship)
    # 1 rework = 0.9, 2 = 0.7, 3 = 0.5, 4+ = 0.3
    rework_factor = max(0.3, 1.0 - (average_rework * 0.2))

    return rework_factor  # Range: 0.3 to 1.0
```

## 📊 Complete Iteration Efficiency Calculation

```python
class IterationEfficiencyCalculator:
    def calculate(self, session, markdown_plan):
        """
        Complete iteration efficiency calculation for complex tasks
        """
        # Core components
        completion_rate = self.calculate_completion_rate(session, markdown_plan)
        quality_score = self.calculate_quality_score(session)
        rework_factor = self.calculate_rework_factor(session, markdown_plan)

        # Additional factors for complex tasks
        momentum_factor = self.calculate_momentum(session)
        coordination_factor = self.calculate_coordination_efficiency(session)

        # Base iteration efficiency
        base_efficiency = completion_rate * quality_score * rework_factor

        # Adjusted for complex task patterns
        adjusted_efficiency = base_efficiency * momentum_factor * coordination_factor

        return {
            'iteration_efficiency': adjusted_efficiency,
            'components': {
                'completion_rate': completion_rate,
                'quality_score': quality_score,
                'rework_factor': rework_factor,
                'momentum_factor': momentum_factor,
                'coordination_factor': coordination_factor
            },
            'interpretation': self.interpret_score(adjusted_efficiency)
        }

    def calculate_momentum(self, session):
        """
        Measures consistent forward progress
        """
        # Extract timestamps and progress markers
        progress_events = extract_progress_markers(session)

        if len(progress_events) < 2:
            return 1.0

        # Calculate time between progress events
        gaps = []
        for i in range(1, len(progress_events)):
            gap = progress_events[i].timestamp - progress_events[i-1].timestamp
            gaps.append(gap)

        # Steady progress = good momentum
        gap_variance = statistics.variance(gaps) if len(gaps) > 1 else 0

        # Lower variance = better momentum
        if gap_variance < 300:  # Less than 5 min variance
            return 1.0  # Excellent momentum
        elif gap_variance < 900:  # Less than 15 min variance
            return 0.9  # Good momentum
        elif gap_variance < 1800:  # Less than 30 min variance
            return 0.8  # Acceptable momentum
        else:
            return 0.7  # Poor momentum

    def calculate_coordination_efficiency(self, session):
        """
        For multi-agent swarm patterns
        """
        agents_used = count_unique_agents(session)

        if agents_used <= 1:
            return 1.0  # No coordination needed

        # Measure coordination overhead
        coordination_messages = count_coordination_patterns(session)
        total_messages = len(session.messages)

        overhead_ratio = coordination_messages / total_messages

        # Optimal coordination: 10-20% of messages
        if 0.1 <= overhead_ratio <= 0.2:
            return 1.0  # Optimal
        elif overhead_ratio < 0.1:
            return 0.9  # Possibly under-coordinated
        elif overhead_ratio <= 0.3:
            return 0.8  # Slightly over-coordinated
        else:
            return 0.6  # Too much coordination overhead

    def interpret_score(self, efficiency):
        """
        What the efficiency score means
        """
        if efficiency >= 0.8:
            return "Excellent - Highly efficient iteration pattern"
        elif efficiency >= 0.6:
            return "Good - Solid progress with acceptable iteration"
        elif efficiency >= 0.4:
            return "Fair - Some inefficiencies but progressing"
        elif efficiency >= 0.2:
            return "Poor - Significant iteration issues"
        else:
            return "Critical - Excessive rework or incomplete execution"
```

## 🎯 Practical Examples

### Example 1: Highly Efficient Session
```python
session_1 = {
    'planned_steps': 12,
    'completed_steps': 11,
    'quality_indicators': ['tests pass', 'requirements met', 'runs successfully'],
    'rework_attempts': [1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1],  # Most steps first try
    'steady_progress': True
}

# Calculation:
completion_rate = 11/12 = 0.917
quality_score = 0.85  # High quality signals
rework_factor = 0.9  # Minimal rework (avg 1.2 attempts)
momentum = 1.0  # Steady progress

iteration_efficiency = 0.917 * 0.85 * 0.9 * 1.0 = 0.702
# Interpretation: "Good - Solid progress with acceptable iteration"
```

### Example 2: Inefficient Session
```python
session_2 = {
    'planned_steps': 12,
    'completed_steps': 5,
    'quality_indicators': ['multiple errors', 'tests failing'],
    'rework_attempts': [5, 4, 6, 3, 7, 0, 0, 0, 0, 0, 0, 0],  # Stuck on early steps
    'steady_progress': False
}

# Calculation:
completion_rate = 5/12 = 0.417
quality_score = 0.3  # Low quality signals
rework_factor = 0.3  # Excessive rework (avg 5 attempts on attempted steps)
momentum = 0.7  # Poor momentum

iteration_efficiency = 0.417 * 0.3 * 0.3 * 0.7 = 0.026
# Interpretation: "Critical - Excessive rework or incomplete execution"
```

## 📈 Optimization Targets

### For Your Complex Tasks
```yaml
Target Metrics:
  Completion Rate: >0.85 (85% of steps completed)
  Quality Score: >0.70 (70% quality indicators present)
  Rework Factor: >0.70 (average <2 attempts per step)
  Momentum: >0.85 (steady progress)
  Coordination: >0.85 (10-20% coordination overhead)

Overall Target: >0.60 iteration efficiency

Optimal Session Pattern:
  - 11-12 planned steps
  - 10-15 messages per step
  - 1-2 attempts per step (some exploration good)
  - Verification after major milestones
  - Total: 110-180 messages
```

## 🔄 Dynamic Adjustment

### Real-time Efficiency Monitoring
```python
class RealTimeEfficiencyMonitor:
    def monitor_session(self, session_id):
        """
        Calculate iteration efficiency during session
        """
        current_step = identify_current_step(session_id)
        completed_steps = count_completed_steps(session_id)
        current_attempts = count_current_step_attempts(session_id)

        # Real-time calculation
        current_efficiency = completed_steps / current_step

        # Predict if intervention needed
        if current_attempts > 4:
            suggest_intervention("Consider alternative approach for this step")

        if current_efficiency < 0.5 and current_step > 6:
            suggest_intervention("Progress slower than optimal, consider simplifying")

        return {
            'current_efficiency': current_efficiency,
            'projected_final': project_final_efficiency(current_efficiency, current_step),
            'health_status': assess_session_health(current_efficiency, current_attempts)
        }
```

## 🎯 Key Takeaways

1. **Iteration Efficiency ≠ Minimal Messages**
   - It's about completing planned work with acceptable quality
   - Some iteration (1-3 attempts) is healthy exploration

2. **Quality Matters More Than Speed**
   - High completion rate with low quality = low efficiency
   - Better to complete fewer steps well than all steps poorly

3. **Rework Penalty is Non-Linear**
   - 1-2 attempts: Minimal penalty (healthy iteration)
   - 3-4 attempts: Moderate penalty (concerning)
   - 5+ attempts: Severe penalty (thrashing)

4. **Your Optimal Range**
   - 0.60-0.85 iteration efficiency for complex tasks
   - Higher than 0.85 might indicate oversimplified tasks
   - Lower than 0.40 indicates systematic issues

This formula specifically optimizes for your thoughtful, planning-oriented workflow where complex tasks are broken into steps and executed systematically. The goal is efficient progress through planned work, not minimal message count.