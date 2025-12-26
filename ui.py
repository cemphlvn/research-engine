"""
Invariant Engine UI - Transparent Epistemological Research Interface

Architecture:
  ORCHESTRATOR (Scientific Method)
      │
      ├── LLM PROVIDER (Hypothesis + Transforms + Adversary)
      ├── EMBEDDING PROVIDER (Measurement Instrument)
      └── STATS ENGINE (Epistemic Court)

The UI reflects this architecture with real-time visualization of the reasoning pipeline.
"""

import os
import sys
import time
from dataclasses import dataclass
from enum import Enum

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, "src")
load_dotenv()

# Page config
st.set_page_config(
    page_title="Invariant",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    /* Remove default padding */
    .block-container { padding-top: 2rem; }

    /* Pipeline step styling */
    .pipeline-step {
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.25rem 0;
        font-family: monospace;
    }
    .step-pending { background: #f0f0f0; color: #888; }
    .step-active { background: #1e88e5; color: white; }
    .step-complete { background: #43a047; color: white; }
    .step-error { background: #e53935; color: white; }

    /* Layer badges */
    .layer-orchestrator { border-left: 4px solid #9c27b0; }
    .layer-llm { border-left: 4px solid #ff9800; }
    .layer-embedding { border-left: 4px solid #2196f3; }
    .layer-stats { border-left: 4px solid #4caf50; }

    /* Metric cards */
    .metric-pass { color: #43a047; }
    .metric-fail { color: #e53935; }

    /* Architecture diagram */
    .arch-box {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    .arch-active { border-color: #1e88e5; background: #e3f2fd; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# STATE & CONFIG
# ============================================================================

@dataclass
class ProviderStatus:
    deepseek: bool = False
    anthropic: bool = False
    openai: bool = False

def get_provider_status() -> ProviderStatus:
    return ProviderStatus(
        deepseek=bool(os.environ.get("DEEPSEEK_API_KEY")),
        anthropic=bool(os.environ.get("ANTHROPIC_API_KEY")),
        openai=bool(os.environ.get("OPENAI_API_KEY")),
    )


# ============================================================================
# ARCHITECTURE VISUALIZATION
# ============================================================================

def render_architecture(active_layer: str = None):
    """Render the system architecture with optional active layer highlight."""

    def box(name: str, items: list, layer: str) -> str:
        active = "arch-active" if active_layer == layer else ""
        items_html = "<br>".join(f"<small>{i}</small>" for i in items)
        return f"""
        <div class="arch-box {active}" style="border-left: 4px solid {'#9c27b0' if layer == 'orchestrator' else '#ff9800' if layer == 'llm' else '#2196f3' if layer == 'embedding' else '#4caf50'}">
            <strong>{name}</strong><br>
            {items_html}
        </div>
        """

    cols = st.columns([1, 3, 1])

    with cols[1]:
        # Orchestrator at top
        st.markdown(box("ORCHESTRATOR", ["Scientific Method", "Wires everything"], "orchestrator"), unsafe_allow_html=True)

        st.markdown("<div style='text-align: center; font-size: 24px;'>│</div>", unsafe_allow_html=True)

        # Three providers below
        sub_cols = st.columns(3)
        with sub_cols[0]:
            st.markdown(box("LLM", ["Hypothesis Gen", "Transform Gen", "Adversary"], "llm"), unsafe_allow_html=True)
        with sub_cols[1]:
            st.markdown(box("EMBEDDING", ["Measurement", "Vectorization"], "embedding"), unsafe_allow_html=True)
        with sub_cols[2]:
            st.markdown(box("STATS", ["Metrics", "Admissibility", "Decision"], "stats"), unsafe_allow_html=True)


def render_pipeline_status(steps: list, current_step: int = -1):
    """Render pipeline steps with status indicators."""

    LAYER_COLORS = {
        "orchestrator": "#9c27b0",
        "llm": "#ff9800",
        "embedding": "#2196f3",
        "stats": "#4caf50",
    }

    for i, (step_name, layer, status) in enumerate(steps):
        if status == "complete":
            icon = "done"
            bg = "#e8f5e9"
            color = "#2e7d32"
        elif status == "active":
            icon = "sync"
            bg = "#e3f2fd"
            color = "#1565c0"
        elif status == "error":
            icon = "error"
            bg = "#ffebee"
            color = "#c62828"
        else:  # pending
            icon = "radio_button_unchecked"
            bg = "#fafafa"
            color = "#9e9e9e"

        layer_color = LAYER_COLORS.get(layer, "#888")

        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 0.4rem 0.8rem; margin: 0.2rem 0;
                    background: {bg}; border-left: 3px solid {layer_color}; border-radius: 4px;">
            <span style="color: {color}; margin-right: 0.5rem;">
                <span class="material-icons" style="font-size: 18px; vertical-align: middle;">{icon}</span>
            </span>
            <span style="color: {color}; font-family: monospace; font-size: 0.9rem;">{step_name}</span>
            <span style="margin-left: auto; color: #888; font-size: 0.75rem;">{layer}</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN UI
# ============================================================================

# Header
st.markdown("# Invariant")
st.markdown("*Epistemological engine: test hypothesis survival under admissible transformations*")

# Sidebar
with st.sidebar:
    st.markdown("## Configuration")

    # Provider status
    status = get_provider_status()

    st.markdown("### Providers")

    providers_available = []
    if status.deepseek:
        providers_available.append("DeepSeek")
        st.markdown("DeepSeek (LLM)")
    if status.anthropic:
        providers_available.append("Anthropic")
        st.markdown("Anthropic (LLM)")
    if status.openai:
        st.markdown("OpenAI (Embeddings)")

    if not providers_available:
        st.warning("No LLM providers configured")
        st.caption("Add DEEPSEEK_API_KEY to .env")

    use_stub = not (status.openai and providers_available)
    if use_stub:
        st.info("Using stub providers (testing mode)")

    st.markdown("---")

    st.markdown("### Thresholds")
    stability_thresh = st.slider("Stability", 0.5, 1.0, 0.80, 0.05)
    separability_thresh = st.slider("Separability", 0.5, 1.0, 0.75, 0.05)
    kl_max = st.slider("Max KL Drift", 0.1, 5.0, 2.0, 0.1)

    st.markdown("---")

    seed = st.number_input("Seed", value=42, min_value=1)


# Main content - two columns
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("## Evaluate Claim")

    claim = st.text_area(
        "Claim",
        value="Justice requires treating similar cases similarly",
        height=80,
        help="The hypothesis to test for invariance"
    )

    positives = st.text_area(
        "Positive Examples",
        value="""Equal treatment under the law is fundamental to justice
Fair trials require impartial judges
Justice means everyone gets what they deserve
A just society protects the vulnerable""",
        height=150,
        help="Examples that should satisfy the claim"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        hypothesis_type = st.selectbox(
            "Type",
            ["concept_invariant", "relation_law", "cluster_structure"],
            index=0,
        )
    with col2:
        run_button = st.button("Run Evaluation", type="primary", use_container_width=True)

with right_col:
    st.markdown("## Architecture")
    render_architecture()


# Divider
st.markdown("---")

# Evaluation execution
if run_button:
    positives_list = [p.strip() for p in positives.split("\n") if p.strip()]

    if not claim or not positives_list:
        st.error("Please enter a claim and at least one positive example")
    else:
        # Create layout for live updates
        pipeline_col, results_col = st.columns([1, 2])

        with pipeline_col:
            st.markdown("### Pipeline")
            pipeline_placeholder = st.empty()

        with results_col:
            st.markdown("### Reasoning Trace")
            trace_container = st.container()

        # Pipeline steps definition
        PIPELINE_STEPS = [
            ("1. Formalize Hypothesis", "llm"),
            ("2. Setup Scoreboard", "orchestrator"),
            ("3. Generate Negatives", "llm"),
            ("4. Embed Originals", "embedding"),
            ("5. Apply Transforms", "llm"),
            ("6. Embed Transformed", "embedding"),
            ("7. Compute Metrics", "stats"),
            ("8. Check Admissibility", "stats"),
            ("9. Apply Decision", "stats"),
            ("10. Package Result", "orchestrator"),
        ]

        # Tracking state
        step_status = ["pending"] * len(PIPELINE_STEPS)
        trace_logs = []

        def update_pipeline(step_idx: int, status: str):
            step_status[step_idx] = status
            steps_with_status = [
                (name, layer, step_status[i])
                for i, (name, layer) in enumerate(PIPELINE_STEPS)
            ]
            with pipeline_placeholder.container():
                for i, (name, layer, s) in enumerate(steps_with_status):
                    if s == "complete":
                        st.markdown(f"✓ **{name}** `{layer}`")
                    elif s == "active":
                        st.markdown(f"◉ **{name}** `{layer}`")
                    elif s == "error":
                        st.markdown(f"✗ **{name}** `{layer}`")
                    else:
                        st.markdown(f"○ {name} `{layer}`")

        def add_trace(layer: str, message: str, data: dict = None):
            ICONS = {"llm": "brain", "embedding": "ruler", "stats": "scale", "orchestrator": "flask"}
            trace_logs.append({"layer": layer, "message": message, "data": data})

            with trace_container:
                for log in trace_logs[-20:]:
                    icon = {"llm": "🧠", "embedding": "📐", "stats": "⚖️", "orchestrator": "🔬"}.get(log["layer"], "•")
                    st.markdown(f"{icon} **{log['layer'].upper()}**: {log['message']}")
                    if log["data"]:
                        for k, v in log["data"].items():
                            if isinstance(v, float):
                                st.caption(f"  → {k}: {v:.4f}")
                            elif isinstance(v, list) and len(v) <= 3:
                                st.caption(f"  → {k}: {v}")
                            elif isinstance(v, list):
                                st.caption(f"  → {k}: [{len(v)} items]")

        # Map EvalStep to pipeline index
        STEP_MAP = {
            "hypothesis_generation": 0,
            "scoreboard_setup": 1,
            "negative_generation": 2,
            "embed_original": 3,
            "apply_transforms": 4,
            "embed_transformed": 5,
            "compute_metrics": 6,
            "check_admissibility": 7,
            "apply_decision": 8,
            "package_result": 9,
        }

        def on_progress(event):
            """Handle progress events from orchestrator."""
            step_idx = STEP_MAP.get(event.step.value, -1)

            if event.status == "start" and step_idx >= 0:
                update_pipeline(step_idx, "active")
            elif event.status == "complete" and step_idx >= 0:
                update_pipeline(step_idx, "complete")
                add_trace(event.layer, event.message, event.data)
            elif event.status == "error" and step_idx >= 0:
                update_pipeline(step_idx, "error")
                add_trace(event.layer, f"ERROR: {event.message}", event.data)

        # Initialize pipeline display
        update_pipeline(-1, "")

        try:
            from invariant.core.orchestrator import Orchestrator, OrchestratorConfig
            from invariant.models.schemas import HypothesisType, EvaluationOutcome

            # Type mapping
            type_map = {
                "concept_invariant": HypothesisType.CONCEPT_INVARIANT,
                "relation_law": HypothesisType.RELATION_LAW,
                "cluster_structure": HypothesisType.CLUSTER_STRUCTURE,
            }

            config = OrchestratorConfig(
                seed=seed,
                stability_threshold=stability_thresh,
                separability_threshold=separability_thresh,
                kl_max=kl_max,
            )

            # Initialize with real or stub providers
            if not use_stub:
                from invariant.providers.openai_provider import OpenAIEmbeddingProvider
                if status.deepseek:
                    # Use semantic provider for proper meaning-preserving evaluation
                    from invariant.providers.semantic import create_semantic_provider_deepseek
                    llm = create_semantic_provider_deepseek()
                    add_trace("orchestrator", "Using SEMANTIC provider (DeepSeek) - real meaning-preserving transforms")
                elif status.anthropic:
                    from invariant.providers.semantic import create_semantic_provider_anthropic
                    llm = create_semantic_provider_anthropic()
                    add_trace("orchestrator", "Using SEMANTIC provider (Anthropic) - real meaning-preserving transforms")
                else:
                    llm = None
                orchestrator = Orchestrator(config=config, llm=llm, embedding=OpenAIEmbeddingProvider())
            else:
                orchestrator = Orchestrator(config=config)
                add_trace("orchestrator", "Using STUB providers - lexical similarity only (testing mode)")

            # Run evaluation
            result = orchestrator.evaluate(
                claim=claim,
                positives=positives_list,
                hypothesis_type=type_map[hypothesis_type],
                on_progress=on_progress,
            )

            # Display result
            st.markdown("---")
            st.markdown("## Result")

            # Outcome banner
            outcome_col, metrics_col = st.columns([1, 1])

            with outcome_col:
                if result.outcome == EvaluationOutcome.DISCOVERY:
                    st.success("## DISCOVERY")
                    if result.discovery:
                        st.markdown(f"**Confidence:** {result.discovery.confidence:.0%}")
                        st.markdown("**Invariant (formal):**")
                        st.code(result.discovery.invariant_formal)
                        st.markdown("**Invariant (plain):**")
                        st.write(result.discovery.invariant_plain)

                elif result.outcome == EvaluationOutcome.FALSIFIED:
                    st.error("## FALSIFIED")
                    if result.falsification:
                        st.markdown(f"**Failure Mode:** `{result.falsification.failure_mode.value}`")
                        st.write(result.falsification.explanation)
                        if result.falsification.suggested_refinements:
                            st.markdown("**Suggestions:**")
                            for s in result.falsification.suggested_refinements[:3]:
                                st.markdown(f"- {s}")

                elif result.outcome == EvaluationOutcome.UNDERDETERMINED:
                    st.warning("## UNDERDETERMINED")
                    if result.underdetermined:
                        st.write(f"**Reason:** {result.underdetermined.reason}")
                        if result.underdetermined.suggested_experiments:
                            st.markdown("**Next Steps:**")
                            for e in result.underdetermined.suggested_experiments[:3]:
                                st.markdown(f"- {e}")
                else:
                    st.info("## INADMISSIBLE")
                    if result.admissibility:
                        st.markdown(f"**Failing Rules:** `{result.admissibility.failing_rules}`")

            with metrics_col:
                st.markdown("### Metrics")

                # Get metrics from result
                metrics_source = result.discovery or result.falsification or result.underdetermined
                if metrics_source and hasattr(metrics_source, "metrics"):
                    for m in metrics_source.metrics.metrics:
                        threshold = {
                            "stability": (stability_thresh, ">="),
                            "separability": (separability_thresh, ">="),
                            "kl_drift": (kl_max, "<="),
                            "mutual_information": (0.5, ">="),
                        }.get(m.name.value)

                        if threshold:
                            thresh_val, op = threshold
                            passed = m.value >= thresh_val if op == ">=" else m.value <= thresh_val
                            status_icon = "✓" if passed else "✗"
                            status_color = "green" if passed else "red"
                        else:
                            status_icon = "•"
                            status_color = "gray"

                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem;
                                    border-bottom: 1px solid #eee;">
                            <span>{m.name.value.replace('_', ' ').title()}</span>
                            <span style="color: {status_color}; font-weight: bold;">
                                {m.value:.3f} {status_icon}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                        if m.ci95:
                            st.caption(f"  95% CI: [{m.ci95[0]:.2f}, {m.ci95[1]:.2f}]")

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


# Footer
st.markdown("---")

# Help section
with st.expander("How It Works"):
    st.markdown("""
    ### The Scientific Method as Code

    1. **Formalize Hypothesis** (LLM) - Convert claim to testable form
    2. **Setup Scoreboard** (Orchestrator) - Define what "winning" means
    3. **Generate Negatives** (LLM Adversary) - Create falsifying examples
    4. **Embed & Transform** (Embedding + LLM) - Measure invariance
    5. **Compute Metrics** (Stats) - Quantify stability, separability, drift
    6. **Check Admissibility** (Stats) - Ensure transforms are valid
    7. **Apply Decision** (Stats) - DISCOVERY, FALSIFIED, or UNDERDETERMINED

    ### Metrics Explained

    | Metric | What it measures | Good value |
    |--------|------------------|------------|
    | **Stability** | Do representations stay similar under transforms? | > 0.8 |
    | **Separability** | Can we distinguish positives from negatives? | > 0.75 |
    | **KL Drift** | How much does the distribution shift? | < 2.0 |
    | **Mutual Information** | How much structure survives transformation? | > 0.5 |

    ### The Key Insight

    **LLMs generate, Stats decide.** Without the stats layer, you get a storytelling engine.
    With it, you get a falsifiable discovery engine.
    """)

st.caption("Invariant Engine v0.2 | Epistemology as Infrastructure")
