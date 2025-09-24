# Claude-SAFLA

**S**ession **A**nalysis **F**or **L**LM **A**lignment

A comprehensive framework for analyzing Claude Code sessions using RLVR (Reinforcement Learning from Verifiable Rewards) to optimize iteration efficiency and user satisfaction in complex, planning-oriented tasks.

## 🎯 Core Hypothesis

> **"Complex thoughtful tasks achieve optimal outcomes when structured as 11-12 step markdown plans, referenced by LLMs, and executed through coordinated multi-agent swarms, with iteration efficiency measured by step completion rate rather than message minimization."**

## 🔬 What is SAFLA?

SAFLA is a data-driven approach to understanding and optimizing LLM interactions, specifically focused on:

- **Analyzing** real Claude Code sessions to identify successful patterns
- **Measuring** iteration efficiency through 6 core metrics
- **Validating** hypotheses about optimal task execution strategies
- **Training** neural models using RLVR for continuous improvement

## 📊 The 6 Core Metrics

1. **Step Completion Rate** - % of defined steps successfully completed
2. **Reasoning Cohesion Score** - Logical flow between execution steps
3. **Context Kelly Criterion** - % of steps with correct context references
4. **Recovery in Long Chains** - % of non-terminal errors resolved
5. **Success with Verification** - % of steps with explicit test/check output
6. **Depth Efficiency** - Success rate in long (>10 steps) vs short (<5 steps) plans

## 🚀 Key Findings

From analyzing 579 Claude Code sessions (77,444 events):

- ✅ **81.6%** of successful sessions use 11-12 step markdown plans
- ✅ **11.0 messages per step** is optimal (not minimal messages)
- ✅ **1.20 depth efficiency** for long plans vs 0.80 for short plans
- ✅ **154.8% recovery rate** shows excellent resilience in complex tasks

## 📈 Iteration Efficiency Formula

```
                    Steps Completed     Quality Score     1
Iteration Efficiency = --------------- × ------------- × ---------------
                     Steps Planned      Max Quality    Rework Factor
```

This formula optimizes for **completing planned work with quality**, not minimizing message count.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/claude-safla.git
cd claude-safla
pip install -r requirements.txt
```

## 📦 Project Structure

```
claude-safla/
├── src/
│   └── rlvr_quality_system.py      # RLVR implementation with 30+ metrics
├── scripts/
│   ├── final_markdown_analysis.py  # Analyze markdown plan sessions
│   └── generate_text_report.py     # Generate visual reports
├── docs/
│   ├── MARKDOWN_PLAN_HYPOTHESIS.md # Core hypothesis details
│   └── ITERATION_EFFICIENCY_FORMULA.md # Mathematical formulation
├── analysis/                       # Output directory for results
└── README.md
```

## 🔧 Usage

### Analyze Your Claude Code Sessions

```python
from src.rlvr_quality_system import RLVRQualitySystem

# Initialize the RLVR system
rlvr = RLVRQualitySystem()

# Analyze sessions from your database
results = rlvr.analyze_sessions('path/to/memory.db')

# Generate quality metrics
metrics = rlvr.calculate_quality_metrics(results)
```

### Run Markdown Plan Analysis

```bash
# Analyze sessions with markdown plan references
python scripts/final_markdown_analysis.py

# Generate visual report
python scripts/generate_text_report.py
```

## 📊 RLVR Quality Metrics

The RLVR system tracks 30+ quantifiable metrics across 7 categories:

- **Task Completion** (20% weight)
- **Code Quality** (15% weight)
- **User Satisfaction** (25% weight)
- **Efficiency** (10% weight)
- **Accuracy** (15% weight)
- **Learning** (10% weight)
- **Collaboration** (5% weight)

## 🎯 Optimal Workflow Pattern

Based on our analysis, the optimal pattern for complex tasks:

1. **Create** 11-12 step markdown plan
2. **Reference** plan to LLM ("Look at plan.md")
3. **Execute** with 10-15 messages per step
4. **Verify** at checkpoints
5. **Iterate** with recovery mechanisms

## 📈 Results Visualization

The analysis generates comprehensive reports showing:

- Step completion rates by plan complexity
- Message efficiency distributions
- Verification patterns
- Recovery rates in long execution chains
- Depth efficiency comparisons

## 🔬 Research Applications

SAFLA can be used for:

- **LLM Alignment** - Training models on successful interaction patterns
- **Workflow Optimization** - Identifying optimal task execution strategies
- **Quality Assurance** - Measuring and improving interaction quality
- **Behavioral Analysis** - Understanding user-LLM collaboration patterns

## 📚 Documentation

- [Markdown Plan Hypothesis](docs/MARKDOWN_PLAN_HYPOTHESIS.md) - Detailed hypothesis and validation
- [Iteration Efficiency Formula](docs/ITERATION_EFFICIENCY_FORMULA.md) - Mathematical framework

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Extending metrics for different task types
- Implementing real-time monitoring
- Building predictive models
- Improving visualization tools

## 📝 Citation

If you use SAFLA in your research, please cite:

```bibtex
@software{claude-safla,
  title = {Claude-SAFLA: Session Analysis For LLM Alignment},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/claude-safla}
}
```

## 🔑 Key Insights

> "Success in Claude Code sessions is driven by clarity of intent, not brevity of interaction."

The framework proves that **thoughtful, planning-oriented workflows** with structured markdown plans achieve better outcomes than minimizing message counts. Iteration efficiency should measure step completion and quality, not speed.

## 📜 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built on analysis of real Claude Code sessions to understand and optimize human-AI collaboration patterns.

---

*SAFLA - Turning session data into actionable insights for better AI alignment*