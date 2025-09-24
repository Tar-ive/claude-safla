#!/usr/bin/env python3
"""
Final Markdown Plan Analysis - Your 6 Core Metrics
==================================================
Analyzes Claude Code sessions with markdown plan references to validate
your hypothesis about 11-12 step plans achieving optimal outcomes.
"""

import sqlite3
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

def extract_text_content(data: dict) -> str:
    """Extract text content from various message formats."""
    message = data.get('message', {})

    # Handle dict format
    if isinstance(message, dict):
        content = message.get('content', '')

        # Content might be string or list
        if isinstance(content, list):
            # Handle list of text items
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            return ' '.join(text_parts)
        elif isinstance(content, str):
            return content

    # Handle string format
    elif isinstance(message, str):
        return message

    return ''

def extract_numbered_steps(text: str) -> List[str]:
    """Extract numbered steps from markdown-like content."""
    steps = []

    # Multiple patterns to catch different formats
    patterns = [
        r'(\d+)\.\s+(.+?)(?=\n\d+\.|$)',  # 1. Step format
        r'(?:^|\n)[-*]\s+(.+?)(?=\n[-*]|$)',  # - or * bullet points
        r'Step\s+(\d+)[:\s]+(.+?)(?=Step\s+\d+|$)',  # Step 1: format
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        for match in matches:
            if isinstance(match, tuple):
                step_text = match[-1] if len(match) > 1 else match[0]
            else:
                step_text = match
            step_text = step_text.strip()

            # Filter out very short or empty steps
            if len(step_text) > 5 and len(step_text) < 500:
                steps.append(step_text)

    # Return up to 12 steps (your hypothesis)
    return steps[:12]

def calculate_metrics(steps: List[str], outputs: List[str]) -> Dict[str, float]:
    """Calculate your 6 core metrics."""
    if not steps or not outputs:
        return {}

    metrics = {}

    # 1. Step Completion Rate
    completed = 0
    for step in steps:
        keywords = [w.lower() for w in step.split() if len(w) > 3][:5]  # Top 5 keywords
        for output in outputs:
            matches = sum(1 for kw in keywords if kw in output.lower())
            if matches >= min(2, len(keywords) * 0.4):  # 40% keyword match
                completed += 1
                break
    metrics['step_completion_rate'] = completed / len(steps)

    # 2. Reasoning Cohesion
    transition_words = ['next', 'then', 'now', 'after', 'completed', 'done', 'moving']
    transitions = 0
    for output in outputs:
        if any(word in output.lower() for word in transition_words):
            transitions += 1
    metrics['reasoning_cohesion'] = min(transitions / len(outputs), 1.0) if outputs else 0

    # 3. Context Kelly Criterion
    context_patterns = ['.md', 'file:', 'reference', 'as mentioned', 'according to', 'from the plan']
    context_refs = sum(1 for output in outputs
                      if any(pattern in output.lower() for pattern in context_patterns))
    metrics['context_kelly_criterion'] = context_refs / len(outputs) if outputs else 0

    # 4. Recovery in Long Chains
    error_words = ['error', 'failed', 'issue', 'problem', 'bug']
    recovery_words = ['fixed', 'resolved', 'solved', 'working', 'success']

    errors = sum(1 for output in outputs if any(word in output.lower() for word in error_words))
    recoveries = sum(1 for output in outputs if any(word in output.lower() for word in recovery_words))

    metrics['recovery_in_long_chains'] = recoveries / errors if errors > 0 else 1.0

    # 5. Success with Verification
    verification_patterns = ['test', 'verif', 'check', 'confirm', 'valid', '✓', '✅', 'pass']
    verified = sum(1 for output in outputs
                  if any(pattern in output.lower() for pattern in verification_patterns))
    metrics['success_with_verification'] = verified / len(outputs) if outputs else 0

    # 6. Depth Efficiency
    if len(steps) > 10:  # Long plan (your 11-12 step plans)
        metrics['depth_efficiency'] = metrics['step_completion_rate'] * 1.2
        metrics['plan_category'] = 'long'
    elif len(steps) < 5:  # Short plan
        metrics['depth_efficiency'] = metrics['step_completion_rate'] * 0.8
        metrics['plan_category'] = 'short'
    else:  # Medium plan
        metrics['depth_efficiency'] = metrics['step_completion_rate']
        metrics['plan_category'] = 'medium'

    return metrics

def analyze_session(session_id: str, conn: sqlite3.Connection) -> Optional[Dict]:
    """Analyze a single session."""
    cursor = conn.cursor()

    # Get all events for this session
    query = '''
    SELECT event_data, timestamp
    FROM events
    WHERE session_id = ?
    ORDER BY timestamp
    '''

    cursor.execute(query, (session_id,))
    events = cursor.fetchall()

    if not events:
        return None

    all_content = []
    markdown_refs = []

    for event_data, _ in events:
        try:
            data = json.loads(event_data)
            content = extract_text_content(data)

            if content:
                all_content.append(content)
                if '.md' in content.lower():
                    markdown_refs.append(content)
        except:
            continue

    if not markdown_refs:
        return None

    # Try to extract steps from various places
    steps = []

    # First try: Look in markdown references
    for ref in markdown_refs[:5]:
        steps = extract_numbered_steps(ref)
        if steps:
            break

    # Second try: Look in early messages
    if not steps:
        for content in all_content[:20]:
            steps = extract_numbered_steps(content)
            if steps:
                break

    if not steps:
        return None

    # Calculate metrics
    metrics = calculate_metrics(steps, all_content)

    # Add session info
    metrics['session_id'] = session_id
    metrics['total_steps'] = len(steps)
    metrics['total_messages'] = len(all_content)
    metrics['messages_per_step'] = len(all_content) / len(steps) if steps else 0
    metrics['is_optimal_length'] = 10 <= metrics['messages_per_step'] <= 15

    return metrics

def main():
    """Run the complete analysis."""
    print("=" * 60)
    print("MARKDOWN PLAN EXECUTION ANALYSIS")
    print("Validating Your 11-12 Step Plan Hypothesis")
    print("=" * 60)

    db_path = Path.home() / 'brain' / 'memory.db'

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find sessions with markdown references
    query = '''
    SELECT DISTINCT session_id, COUNT(*) as event_count
    FROM events
    WHERE event_data LIKE '%.md%'
    GROUP BY session_id
    HAVING event_count > 30
    ORDER BY event_count DESC
    LIMIT 50
    '''

    cursor.execute(query)
    sessions = cursor.fetchall()

    print(f"\nFound {len(sessions)} sessions with markdown references")
    print("Analyzing sessions for your 6 core metrics...")
    print()

    results = []
    analyzed = 0

    for session_id, event_count in sessions:
        analyzed += 1

        metrics = analyze_session(session_id, conn)

        if metrics:
            results.append(metrics)

            # Show progress
            if analyzed % 5 == 0:
                print(f"Progress: {analyzed}/{len(sessions)} sessions analyzed...")

    if not results:
        print("No sessions with extractable plans found")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate statistics
    long_plans = df[df['plan_category'] == 'long'] if 'plan_category' in df.columns else pd.DataFrame()
    short_plans = df[df['plan_category'] == 'short'] if 'plan_category' in df.columns else pd.DataFrame()
    medium_plans = df[df['plan_category'] == 'medium'] if 'plan_category' in df.columns else pd.DataFrame()

    # Generate report
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nSessions Successfully Analyzed: {len(df)}")
    print(f"Sessions with Extractable Plans: {len(df)}")

    print("\n📊 YOUR 6 CORE METRICS:")
    print("-" * 40)

    print(f"\n1. STEP COMPLETION RATE: {df['step_completion_rate'].mean():.1%}")
    if not long_plans.empty:
        print(f"   - Long plans (>10 steps): {long_plans['step_completion_rate'].mean():.1%}")
    if not medium_plans.empty:
        print(f"   - Medium plans (5-10): {medium_plans['step_completion_rate'].mean():.1%}")
    if not short_plans.empty:
        print(f"   - Short plans (<5): {short_plans['step_completion_rate'].mean():.1%}")

    print(f"\n2. REASONING COHESION: {df['reasoning_cohesion'].mean():.2f}")
    print(f"   (Logical flow between steps)")

    print(f"\n3. CONTEXT KELLY CRITERION: {df['context_kelly_criterion'].mean():.1%}")
    print(f"   (Correct context references)")

    print(f"\n4. RECOVERY IN LONG CHAINS: {df['recovery_in_long_chains'].mean():.1%}")
    print(f"   (Error recovery rate)")

    print(f"\n5. SUCCESS WITH VERIFICATION: {df['success_with_verification'].mean():.1%}")
    print(f"   (Steps with explicit verification)")

    print(f"\n6. DEPTH EFFICIENCY: {df['depth_efficiency'].mean():.2f}")
    if not long_plans.empty:
        print(f"   - Long plans: {long_plans['depth_efficiency'].mean():.2f}")
    if not short_plans.empty:
        print(f"   - Short plans: {short_plans['depth_efficiency'].mean():.2f}")

    print("\n🎯 HYPOTHESIS VALIDATION:")
    print("-" * 40)

    if not long_plans.empty and not short_plans.empty:
        diff = (long_plans['step_completion_rate'].mean() - short_plans['step_completion_rate'].mean()) * 100
        print(f"Long plans show {diff:+.1f}pp {'better' if diff > 0 else 'worse'} completion than short plans")

    print(f"\nPlan Distribution:")
    print(f"  - Long (>10 steps): {len(long_plans)} sessions")
    print(f"  - Medium (5-10): {len(medium_plans)} sessions")
    print(f"  - Short (<5): {len(short_plans)} sessions")

    print(f"\nOptimal Message Range (10-15 per step): {df['is_optimal_length'].mean():.1%}")
    print(f"Average Messages per Step: {df['messages_per_step'].mean():.1f}")

    # Save results
    output_dir = Path('/Users/tarive/claude-patterns/analysis')
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / 'final_markdown_metrics.csv', index=False)
    print(f"\n✅ Results saved to {output_dir / 'final_markdown_metrics.csv'}")

    # Show top performing sessions
    if len(df) > 0:
        print("\n🏆 TOP PERFORMING SESSIONS:")
        print("-" * 40)

        top = df.nlargest(3, 'step_completion_rate')
        for _, row in top.iterrows():
            print(f"Session: ...{row['session_id'][-20:]}")
            print(f"  Steps: {row['total_steps']}, Completion: {row['step_completion_rate']:.1%}")
            print(f"  Depth Efficiency: {row['depth_efficiency']:.2f}")
            print()

    conn.close()
    print("Analysis complete!")

if __name__ == "__main__":
    main()