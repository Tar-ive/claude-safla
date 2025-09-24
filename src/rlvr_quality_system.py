#!/usr/bin/env python3
"""
RLVR (Reinforcement Learning from Verifiable Rewards) Quality System
Quantifiable metrics for Claude conversation quality and neural training
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
import math

@dataclass
class QualityMetrics:
    """Quantifiable quality metrics for RLVR"""

    # Task Completion Metrics
    task_completion_rate: float = 0.0  # % of tasks marked complete
    time_to_completion: float = 0.0    # Average time in minutes
    first_attempt_success: float = 0.0  # % solved on first try
    retry_count: float = 0.0            # Average retries needed

    # Code Quality Metrics
    syntax_error_rate: float = 0.0     # % of code with syntax errors
    runtime_error_rate: float = 0.0    # % of code with runtime errors
    test_pass_rate: float = 0.0        # % of generated tests passing
    lint_score: float = 0.0            # Average linting score (0-100)

    # User Satisfaction Metrics
    explicit_positive_feedback: float = 0.0  # Count of "thanks", "perfect", etc.
    explicit_negative_feedback: float = 0.0  # Count of "error", "wrong", etc.
    implicit_satisfaction: float = 0.0       # Derived from interaction patterns
    session_continuation_rate: float = 0.0   # % users who continue after response

    # Efficiency Metrics
    token_efficiency: float = 0.0      # Task completion / tokens used
    response_time: float = 0.0         # Average response generation time
    context_utilization: float = 0.0   # % of context effectively used
    redundancy_rate: float = 0.0       # % of redundant explanations

    # Accuracy Metrics
    factual_accuracy: float = 0.0      # % of verifiable facts correct
    tool_selection_accuracy: float = 0.0  # % correct tool choices
    command_success_rate: float = 0.0     # % of bash commands that succeed
    file_operation_success: float = 0.0   # % of file ops without errors

    # Learning Metrics
    pattern_recognition: float = 0.0    # Ability to recognize similar problems
    improvement_over_time: float = 0.0  # Quality trend over sessions
    adaptation_speed: float = 0.0       # How quickly learns user preferences
    knowledge_retention: float = 0.0    # Remembers project context

    # Collaboration Metrics
    clarification_rate: float = 0.0     # % times asks for clarification
    assumption_accuracy: float = 0.0    # % correct assumptions
    context_awareness: float = 0.0      # Uses relevant context appropriately
    proactive_suggestions: float = 0.0  # Helpful suggestions made

    def to_reward_vector(self) -> np.ndarray:
        """Convert metrics to RLVR reward vector"""
        return np.array([
            self.task_completion_rate,
            self.first_attempt_success,
            1.0 - self.syntax_error_rate,
            1.0 - self.runtime_error_rate,
            self.test_pass_rate,
            self.explicit_positive_feedback / (self.explicit_positive_feedback + self.explicit_negative_feedback + 1e-8),
            self.session_continuation_rate,
            self.token_efficiency,
            self.factual_accuracy,
            self.tool_selection_accuracy,
            self.command_success_rate,
            self.pattern_recognition,
            self.improvement_over_time,
            self.context_awareness
        ])


class RLVRQualityAnalyzer:
    """Analyzes conversation quality for RLVR training"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else Path.home() / "brain" / "memory_cleaned.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # RLVR reward weights (learned or configured)
        self.reward_weights = {
            'task_completion': 0.20,
            'code_quality': 0.15,
            'user_satisfaction': 0.25,
            'efficiency': 0.10,
            'accuracy': 0.15,
            'learning': 0.10,
            'collaboration': 0.05
        }

    def calculate_task_completion_metrics(self, session_id: str) -> Dict[str, float]:
        """Calculate task completion metrics for a session"""
        metrics = {}

        # Get all events for session
        self.cursor.execute("""
            SELECT event_type, event_data, timestamp
            FROM events
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))

        events = self.cursor.fetchall()

        # Analyze task completion patterns
        tasks_started = 0
        tasks_completed = 0
        first_attempt_success = 0
        retry_sequences = []

        for i, (event_type, event_data, timestamp) in enumerate(events):
            try:
                data = json.loads(event_data) if isinstance(event_data, str) else event_data
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)
                content_lower = content.lower()

                # Detect task starts
                task_indicators = ['please', 'can you', 'help me', 'i need', 'create', 'build', 'fix']
                if event_type == 'user' and any(ind in content_lower for ind in task_indicators):
                    tasks_started += 1

                # Detect task completions
                completion_indicators = ['done', 'complete', 'finished', 'works', 'thank', 'perfect']
                if event_type == 'user' and any(ind in content_lower for ind in completion_indicators):
                    tasks_completed += 1

                    # Check if this was first attempt (no error messages in between)
                    if i > 0:
                        prev_content = str(events[i-1][1]).lower()
                        if 'error' not in prev_content and 'fail' not in prev_content:
                            first_attempt_success += 1

                # Detect retries
                if 'try again' in content_lower or 'retry' in content_lower:
                    retry_sequences.append(1)

            except (json.JSONDecodeError, AttributeError):
                continue

        metrics['task_completion_rate'] = tasks_completed / max(tasks_started, 1)
        metrics['first_attempt_success'] = first_attempt_success / max(tasks_completed, 1)
        metrics['retry_count'] = len(retry_sequences)

        # Calculate time to completion (in minutes)
        if len(events) >= 2 and events[0][2] and events[-1][2]:
            try:
                first_time_str = events[0][2].replace('Z', '+00:00') if events[0][2] else None
                last_time_str = events[-1][2].replace('Z', '+00:00') if events[-1][2] else None

                if first_time_str and last_time_str:
                    first_time = datetime.fromisoformat(first_time_str)
                    last_time = datetime.fromisoformat(last_time_str)
                    metrics['time_to_completion'] = (last_time - first_time).total_seconds() / 60
                else:
                    metrics['time_to_completion'] = 0
            except (ValueError, AttributeError):
                metrics['time_to_completion'] = 0
        else:
            metrics['time_to_completion'] = 0

        return metrics

    def calculate_code_quality_metrics(self, session_id: str) -> Dict[str, float]:
        """Calculate code quality metrics"""
        metrics = {}

        self.cursor.execute("""
            SELECT event_type, event_data
            FROM events
            WHERE session_id = ?
        """, (session_id,))

        syntax_errors = 0
        runtime_errors = 0
        test_mentions = 0
        test_passes = 0
        code_blocks = 0

        for event_type, event_data in self.cursor.fetchall():
            try:
                data = json.loads(event_data) if isinstance(event_data, str) else event_data
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)
                content_lower = content.lower()

                # Count code blocks
                if '```' in content:
                    code_blocks += content.count('```') // 2

                # Detect errors
                if 'syntaxerror' in content_lower or 'syntax error' in content_lower:
                    syntax_errors += 1
                if 'error' in content_lower and 'runtime' in content_lower:
                    runtime_errors += 1
                if 'traceback' in content_lower:
                    runtime_errors += 1

                # Detect test results
                if 'test' in content_lower and 'pass' in content_lower:
                    test_passes += 1
                    test_mentions += 1
                elif 'test' in content_lower and 'fail' in content_lower:
                    test_mentions += 1

            except (json.JSONDecodeError, AttributeError):
                continue

        metrics['syntax_error_rate'] = syntax_errors / max(code_blocks, 1)
        metrics['runtime_error_rate'] = runtime_errors / max(code_blocks, 1)
        metrics['test_pass_rate'] = test_passes / max(test_mentions, 1)
        metrics['lint_score'] = 100 * (1 - (syntax_errors + runtime_errors) / max(code_blocks * 2, 1))

        return metrics

    def calculate_user_satisfaction_metrics(self, session_id: str) -> Dict[str, float]:
        """Calculate user satisfaction metrics"""
        metrics = {}

        self.cursor.execute("""
            SELECT event_type, event_data
            FROM events
            WHERE session_id = ? AND event_type = 'user'
        """, (session_id,))

        positive_count = 0
        negative_count = 0
        total_user_messages = 0
        continuation_signals = 0

        # Positive indicators with weights
        positive_patterns = {
            'perfect': 3,
            'excellent': 3,
            'great work': 3,
            'thanks so much': 2,
            'thank you': 2,
            'works': 2,
            'helpful': 2,
            'good': 1,
            'nice': 1,
            'ok': 0.5
        }

        # Negative indicators with weights
        negative_patterns = {
            'terrible': -3,
            'awful': -3,
            'completely wrong': -3,
            'doesn\'t work': -2,
            'broken': -2,
            'error': -2,
            'failed': -2,
            'wrong': -1,
            'issue': -1,
            'problem': -1
        }

        for event_type, event_data in self.cursor.fetchall():
            try:
                data = json.loads(event_data) if isinstance(event_data, str) else event_data
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)
                content_lower = content.lower()

                total_user_messages += 1

                # Count weighted positive feedback
                for pattern, weight in positive_patterns.items():
                    if pattern in content_lower:
                        positive_count += weight

                # Count weighted negative feedback
                for pattern, weight in negative_patterns.items():
                    if pattern in content_lower:
                        negative_count += abs(weight)

                # Detect continuation (user keeps engaging)
                if len(content) > 20:  # Substantial message
                    continuation_signals += 1

            except (json.JSONDecodeError, AttributeError):
                continue

        metrics['explicit_positive_feedback'] = positive_count
        metrics['explicit_negative_feedback'] = negative_count

        # Calculate implicit satisfaction (based on engagement)
        if total_user_messages > 0:
            metrics['implicit_satisfaction'] = (positive_count - negative_count) / total_user_messages
            metrics['session_continuation_rate'] = continuation_signals / total_user_messages
        else:
            metrics['implicit_satisfaction'] = 0
            metrics['session_continuation_rate'] = 0

        return metrics

    def calculate_efficiency_metrics(self, session_id: str) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        metrics = {}

        self.cursor.execute("""
            SELECT event_type, event_data
            FROM events
            WHERE session_id = ?
        """, (session_id,))

        total_tokens = 0
        useful_content = 0
        redundant_content = 0
        response_count = 0

        seen_explanations = set()

        for event_type, event_data in self.cursor.fetchall():
            try:
                data = json.loads(event_data) if isinstance(event_data, str) else event_data
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)

                if event_type == 'assistant':
                    response_count += 1
                    # Estimate tokens (rough approximation)
                    total_tokens += len(content.split())

                    # Check for redundancy
                    content_hash = hash(content[:100])  # First 100 chars
                    if content_hash in seen_explanations:
                        redundant_content += 1
                    else:
                        seen_explanations.add(content_hash)
                        useful_content += 1

            except (json.JSONDecodeError, AttributeError):
                continue

        # Get task completion for efficiency calculation
        task_metrics = self.calculate_task_completion_metrics(session_id)

        metrics['token_efficiency'] = task_metrics['task_completion_rate'] / max(total_tokens / 1000, 1)
        metrics['context_utilization'] = useful_content / max(response_count, 1)
        metrics['redundancy_rate'] = redundant_content / max(response_count, 1)
        metrics['response_time'] = total_tokens / max(response_count, 1)  # Avg tokens per response

        return metrics

    def calculate_accuracy_metrics(self, session_id: str) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        metrics = {}

        self.cursor.execute("""
            SELECT event_type, event_data
            FROM events
            WHERE session_id = ?
        """, (session_id,))

        tool_uses = 0
        tool_successes = 0
        commands_run = 0
        commands_succeeded = 0
        file_ops = 0
        file_ops_succeeded = 0

        for event_type, event_data in self.cursor.fetchall():
            try:
                data = json.loads(event_data) if isinstance(event_data, str) else event_data
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)
                content_lower = content.lower()

                # Detect tool usage
                if event_type == 'tool_use':
                    tool_uses += 1
                    # Simple heuristic: if no error follows, it succeeded
                    if 'error' not in content_lower:
                        tool_successes += 1

                # Detect command execution
                if 'bash' in content_lower or '$' in content:
                    commands_run += 1
                    if 'error' not in content_lower and 'fail' not in content_lower:
                        commands_succeeded += 1

                # Detect file operations
                if any(op in content_lower for op in ['write', 'read', 'create', 'delete', 'edit']):
                    file_ops += 1
                    if 'error' not in content_lower and 'permission' not in content_lower:
                        file_ops_succeeded += 1

            except (json.JSONDecodeError, AttributeError):
                continue

        metrics['tool_selection_accuracy'] = tool_successes / max(tool_uses, 1)
        metrics['command_success_rate'] = commands_succeeded / max(commands_run, 1)
        metrics['file_operation_success'] = file_ops_succeeded / max(file_ops, 1)
        metrics['factual_accuracy'] = 0.8  # Placeholder - would need fact checking

        return metrics

    def calculate_learning_metrics(self, session_id: str) -> Dict[str, float]:
        """Calculate learning and adaptation metrics"""
        metrics = {}

        # Get session timeline
        self.cursor.execute("""
            SELECT start_time, message_count
            FROM sessions
            WHERE session_id = ?
        """, (session_id,))

        session_data = self.cursor.fetchone()
        if not session_data:
            return {
                'pattern_recognition': 0,
                'improvement_over_time': 0,
                'adaptation_speed': 0,
                'knowledge_retention': 0
            }

        # Pattern recognition: Check if similar problems are solved faster
        self.cursor.execute("""
            SELECT event_data
            FROM events
            WHERE session_id = ? AND event_type = 'assistant'
        """, (session_id,))

        responses = self.cursor.fetchall()
        pattern_matches = 0
        unique_patterns = set()

        for response in responses:
            try:
                data = json.loads(response[0]) if isinstance(response[0], str) else response[0]
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)

                # Extract code patterns
                if '```' in content:
                    code_block = content.split('```')[1] if len(content.split('```')) > 1 else ''
                    pattern_hash = hash(code_block[:50])  # First 50 chars of code

                    if pattern_hash in unique_patterns:
                        pattern_matches += 1
                    else:
                        unique_patterns.add(pattern_hash)

            except (json.JSONDecodeError, AttributeError):
                continue

        metrics['pattern_recognition'] = pattern_matches / max(len(responses), 1)

        # Improvement over time: Compare early vs late response quality
        if len(responses) > 10:
            early_quality = self._assess_response_quality(responses[:5])
            late_quality = self._assess_response_quality(responses[-5:])
            metrics['improvement_over_time'] = (late_quality - early_quality) / max(early_quality, 1)
        else:
            metrics['improvement_over_time'] = 0

        # Adaptation speed: How quickly adjusts to user style
        metrics['adaptation_speed'] = 1.0 / max(session_data[1] / 10, 1)  # Inverse of messages needed

        # Knowledge retention: References to earlier context
        context_references = 0
        for response in responses:
            try:
                data = json.loads(response[0]) if isinstance(response[0], str) else response[0]
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)

                if any(ref in content.lower() for ref in ['earlier', 'previously', 'before', 'as mentioned']):
                    context_references += 1

            except (json.JSONDecodeError, AttributeError):
                continue

        metrics['knowledge_retention'] = context_references / max(len(responses), 1)

        return metrics

    def calculate_collaboration_metrics(self, session_id: str) -> Dict[str, float]:
        """Calculate collaboration quality metrics"""
        metrics = {}

        self.cursor.execute("""
            SELECT event_type, event_data
            FROM events
            WHERE session_id = ? AND event_type = 'assistant'
        """, (session_id,))

        clarifications = 0
        assumptions = 0
        correct_assumptions = 0
        proactive_suggestions = 0
        context_uses = 0
        total_responses = 0

        for event_type, event_data in self.cursor.fetchall():
            try:
                data = json.loads(event_data) if isinstance(event_data, str) else event_data
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)
                content_lower = content.lower()

                total_responses += 1

                # Detect clarifications
                clarification_patterns = ['could you clarify', 'do you mean', 'to confirm', 'just to be sure']
                if any(pattern in content_lower for pattern in clarification_patterns):
                    clarifications += 1

                # Detect assumptions
                assumption_patterns = ['i assume', 'i\'ll assume', 'assuming', 'i believe you mean']
                if any(pattern in content_lower for pattern in assumption_patterns):
                    assumptions += 1
                    # Simple heuristic: if user doesn't correct, assumption was right
                    correct_assumptions += 0.7  # 70% success rate estimate

                # Detect proactive suggestions
                suggestion_patterns = ['you might also', 'consider', 'alternatively', 'i suggest', 'recommendation']
                if any(pattern in content_lower for pattern in suggestion_patterns):
                    proactive_suggestions += 1

                # Detect context awareness
                context_patterns = ['based on', 'given that', 'since you', 'in your project']
                if any(pattern in content_lower for pattern in context_patterns):
                    context_uses += 1

            except (json.JSONDecodeError, AttributeError):
                continue

        metrics['clarification_rate'] = clarifications / max(total_responses, 1)
        metrics['assumption_accuracy'] = correct_assumptions / max(assumptions, 1)
        metrics['context_awareness'] = context_uses / max(total_responses, 1)
        metrics['proactive_suggestions'] = proactive_suggestions / max(total_responses, 1)

        return metrics

    def _assess_response_quality(self, responses: List) -> float:
        """Helper to assess quality of responses"""
        quality_score = 0
        for response in responses:
            try:
                data = json.loads(response[0]) if isinstance(response[0], str) else response[0]
                content = str(data.get('content', '')) if isinstance(data, dict) else str(data)

                # Simple quality heuristics
                if '```' in content:  # Contains code
                    quality_score += 1
                if len(content) > 100:  # Substantial response
                    quality_score += 0.5
                if 'error' not in content.lower():  # No errors
                    quality_score += 0.5

            except (json.JSONDecodeError, AttributeError):
                continue

        return quality_score / max(len(responses), 1)

    def calculate_rlvr_reward(self, session_id: str) -> Tuple[float, QualityMetrics]:
        """Calculate comprehensive RLVR reward for a session"""

        # Calculate all metrics
        metrics = QualityMetrics()

        # Task completion
        task_metrics = self.calculate_task_completion_metrics(session_id)
        metrics.task_completion_rate = task_metrics['task_completion_rate']
        metrics.time_to_completion = task_metrics['time_to_completion']
        metrics.first_attempt_success = task_metrics['first_attempt_success']
        metrics.retry_count = task_metrics['retry_count']

        # Code quality
        code_metrics = self.calculate_code_quality_metrics(session_id)
        metrics.syntax_error_rate = code_metrics['syntax_error_rate']
        metrics.runtime_error_rate = code_metrics['runtime_error_rate']
        metrics.test_pass_rate = code_metrics['test_pass_rate']
        metrics.lint_score = code_metrics['lint_score']

        # User satisfaction
        satisfaction_metrics = self.calculate_user_satisfaction_metrics(session_id)
        metrics.explicit_positive_feedback = satisfaction_metrics['explicit_positive_feedback']
        metrics.explicit_negative_feedback = satisfaction_metrics['explicit_negative_feedback']
        metrics.implicit_satisfaction = satisfaction_metrics['implicit_satisfaction']
        metrics.session_continuation_rate = satisfaction_metrics['session_continuation_rate']

        # Efficiency
        efficiency_metrics = self.calculate_efficiency_metrics(session_id)
        metrics.token_efficiency = efficiency_metrics['token_efficiency']
        metrics.response_time = efficiency_metrics['response_time']
        metrics.context_utilization = efficiency_metrics['context_utilization']
        metrics.redundancy_rate = efficiency_metrics['redundancy_rate']

        # Accuracy
        accuracy_metrics = self.calculate_accuracy_metrics(session_id)
        metrics.factual_accuracy = accuracy_metrics['factual_accuracy']
        metrics.tool_selection_accuracy = accuracy_metrics['tool_selection_accuracy']
        metrics.command_success_rate = accuracy_metrics['command_success_rate']
        metrics.file_operation_success = accuracy_metrics['file_operation_success']

        # Learning
        learning_metrics = self.calculate_learning_metrics(session_id)
        metrics.pattern_recognition = learning_metrics['pattern_recognition']
        metrics.improvement_over_time = learning_metrics['improvement_over_time']
        metrics.adaptation_speed = learning_metrics['adaptation_speed']
        metrics.knowledge_retention = learning_metrics['knowledge_retention']

        # Collaboration
        collab_metrics = self.calculate_collaboration_metrics(session_id)
        metrics.clarification_rate = collab_metrics['clarification_rate']
        metrics.assumption_accuracy = collab_metrics['assumption_accuracy']
        metrics.context_awareness = collab_metrics['context_awareness']
        metrics.proactive_suggestions = collab_metrics['proactive_suggestions']

        # Calculate weighted RLVR reward
        reward_components = {
            'task_completion': (metrics.task_completion_rate + metrics.first_attempt_success) / 2,
            'code_quality': (metrics.test_pass_rate + metrics.lint_score / 100) / 2,
            'user_satisfaction': (metrics.implicit_satisfaction + 1) / 2,  # Normalize to 0-1
            'efficiency': metrics.token_efficiency,
            'accuracy': (metrics.tool_selection_accuracy + metrics.command_success_rate) / 2,
            'learning': (metrics.pattern_recognition + metrics.knowledge_retention) / 2,
            'collaboration': (metrics.context_awareness + metrics.assumption_accuracy) / 2
        }

        # Calculate weighted reward
        total_reward = sum(
            self.reward_weights[key] * value
            for key, value in reward_components.items()
        )

        return total_reward, metrics

    def analyze_all_sessions(self) -> Dict[str, Any]:
        """Analyze all sessions and generate RLVR training data"""

        self.cursor.execute("SELECT session_id FROM sessions")
        session_ids = [row[0] for row in self.cursor.fetchall()]

        all_rewards = []
        all_metrics = []
        session_rewards = {}

        print(f"Analyzing {len(session_ids)} sessions for RLVR...")

        for i, session_id in enumerate(session_ids):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(session_ids)}")

            reward, metrics = self.calculate_rlvr_reward(session_id)
            all_rewards.append(reward)
            all_metrics.append(metrics)
            session_rewards[session_id] = reward

        # Calculate statistics
        rewards_array = np.array(all_rewards)

        stats = {
            'total_sessions': len(session_ids),
            'mean_reward': float(np.mean(rewards_array)),
            'std_reward': float(np.std(rewards_array)),
            'min_reward': float(np.min(rewards_array)),
            'max_reward': float(np.max(rewards_array)),
            'median_reward': float(np.median(rewards_array)),
            'percentiles': {
                '25th': float(np.percentile(rewards_array, 25)),
                '50th': float(np.percentile(rewards_array, 50)),
                '75th': float(np.percentile(rewards_array, 75)),
                '90th': float(np.percentile(rewards_array, 90)),
                '95th': float(np.percentile(rewards_array, 95))
            },
            'reward_distribution': {
                'very_low': int(np.sum(rewards_array < 0.2)),
                'low': int(np.sum((rewards_array >= 0.2) & (rewards_array < 0.4))),
                'medium': int(np.sum((rewards_array >= 0.4) & (rewards_array < 0.6))),
                'high': int(np.sum((rewards_array >= 0.6) & (rewards_array < 0.8))),
                'very_high': int(np.sum(rewards_array >= 0.8))
            },
            'top_sessions': sorted(session_rewards.items(), key=lambda x: x[1], reverse=True)[:10],
            'bottom_sessions': sorted(session_rewards.items(), key=lambda x: x[1])[:10]
        }

        # Save RLVR training data
        rlvr_data = {
            'rewards': session_rewards,
            'metrics': [m.__dict__ for m in all_metrics],
            'statistics': stats,
            'reward_weights': self.reward_weights,
            'generated_at': datetime.now().isoformat()
        }

        output_path = Path("claude-patterns/rlvr_training_data.json")
        with open(output_path, 'w') as f:
            json.dump(rlvr_data, f, indent=2)

        print(f"\n✅ RLVR analysis complete!")
        print(f"📄 Saved to: {output_path}")

        return stats

    def generate_rlvr_report(self, stats: Dict[str, Any]):
        """Generate comprehensive RLVR quality report"""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RLVR Quality Metrics Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 40px 0; }}
        h1 {{ font-size: 48px; margin: 0; }}
        .reward-score {{ font-size: 72px; font-weight: bold; color: #FFD700; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 40px 0; }}
        .metric-card {{ background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; backdrop-filter: blur(10px); }}
        .metric-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #FFD700; }}
        .metric-label {{ font-size: 14px; opacity: 0.8; }}
        .distribution-chart {{ background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; margin: 20px 0; }}
        .bar {{ height: 30px; background: linear-gradient(90deg, #FFD700, #FFA500); border-radius: 5px; margin: 5px 0; }}
        .quality-indicator {{ display: inline-block; width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }}
        .high-quality {{ background: #4CAF50; }}
        .medium-quality {{ background: #FFC107; }}
        .low-quality {{ background: #F44336; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 RLVR Quality Metrics Analysis</h1>
            <div class="reward-score">{stats['mean_reward']:.3f}</div>
            <div style="font-size: 24px;">Average Reward Score</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">📊 Statistical Overview</div>
                <div class="metric-value">{stats['total_sessions']}</div>
                <div class="metric-label">Total Sessions Analyzed</div>
                <hr style="opacity: 0.3;">
                <div>Std Dev: {stats['std_reward']:.3f}</div>
                <div>Min: {stats['min_reward']:.3f}</div>
                <div>Max: {stats['max_reward']:.3f}</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">📈 Percentile Distribution</div>
                <div>95th: {stats['percentiles']['95th']:.3f}</div>
                <div>90th: {stats['percentiles']['90th']:.3f}</div>
                <div>75th: {stats['percentiles']['75th']:.3f}</div>
                <div>50th: {stats['percentiles']['50th']:.3f}</div>
                <div>25th: {stats['percentiles']['25th']:.3f}</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">🎨 Quality Distribution</div>
                <div><span class="quality-indicator high-quality"></span>Very High (≥0.8): {stats['reward_distribution']['very_high']}</div>
                <div><span class="quality-indicator high-quality"></span>High (0.6-0.8): {stats['reward_distribution']['high']}</div>
                <div><span class="quality-indicator medium-quality"></span>Medium (0.4-0.6): {stats['reward_distribution']['medium']}</div>
                <div><span class="quality-indicator low-quality"></span>Low (0.2-0.4): {stats['reward_distribution']['low']}</div>
                <div><span class="quality-indicator low-quality"></span>Very Low (<0.2): {stats['reward_distribution']['very_low']}</div>
            </div>
        </div>

        <div class="distribution-chart">
            <h2>Reward Distribution</h2>
            <div>Very High: <div class="bar" style="width: {stats['reward_distribution']['very_high']/stats['total_sessions']*100}%"></div></div>
            <div>High: <div class="bar" style="width: {stats['reward_distribution']['high']/stats['total_sessions']*100}%"></div></div>
            <div>Medium: <div class="bar" style="width: {stats['reward_distribution']['medium']/stats['total_sessions']*100}%"></div></div>
            <div>Low: <div class="bar" style="width: {stats['reward_distribution']['low']/stats['total_sessions']*100}%"></div></div>
            <div>Very Low: <div class="bar" style="width: {stats['reward_distribution']['very_low']/stats['total_sessions']*100}%"></div></div>
        </div>

        <div class="metric-card">
            <h2>🏆 Top Performing Sessions</h2>
            <ol>
                {''.join(f"<li>{sid[:8]}... - Reward: {reward:.3f}</li>" for sid, reward in stats['top_sessions'][:5])}
            </ol>
        </div>

        <div class="metric-card">
            <h2>⚠️ Sessions Needing Improvement</h2>
            <ol>
                {''.join(f"<li>{sid[:8]}... - Reward: {reward:.3f}</li>" for sid, reward in stats['bottom_sessions'][:5])}
            </ol>
        </div>
    </div>
</body>
</html>"""

        output_path = Path("claude-patterns/rlvr_quality_report.html")
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"📊 RLVR report saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    print("=" * 50)
    print("RLVR QUALITY ANALYSIS SYSTEM")
    print("=" * 50)

    # Initialize analyzer
    analyzer = RLVRQualityAnalyzer()

    # Analyze all sessions
    stats = analyzer.analyze_all_sessions()

    # Generate report
    report_path = analyzer.generate_rlvr_report(stats)

    print("\n📊 RLVR Analysis Summary:")
    print(f"  Mean Reward: {stats['mean_reward']:.3f}")
    print(f"  Std Dev: {stats['std_reward']:.3f}")
    print(f"  Range: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]")

    print("\n🎯 Quality Distribution:")
    total = stats['total_sessions']
    for quality, count in stats['reward_distribution'].items():
        percentage = count / total * 100
        print(f"  {quality.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

    print(f"\n✅ Analysis complete!")
    print(f"📄 Training data: claude-patterns/rlvr_training_data.json")
    print(f"📊 Report: {report_path}")
    print(f"\n🚀 Ready for RLVR training with quantifiable quality metrics!")