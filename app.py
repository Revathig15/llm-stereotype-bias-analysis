
import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="LLM Stereotype Bias Analysis",
    page_icon="⚖️",
    layout="wide"
)

FOLDER = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    files = {
        "Qwen Baseline": "Qwen_Baseline_Corrected.csv",
        "Qwen Debiased": "Qwen_Debiased_Corrected.csv",
        "Gemma Baseline": "Gemma_Baseline_300.csv",
        "Gemma Debiased": "Gemma_Debiased_300.csv"
    }

    data = {
        name: pd.read_csv(os.path.join(FOLDER, filename))
        for name, filename in files.items()
    }

    qwen_scores = pd.read_csv(
        os.path.join(FOLDER, "Qwen_Corrected_Overall_Scores.csv")
    )

    gemma_scores = pd.read_csv(
        os.path.join(FOLDER, "Gemma_Overall_Bias_Scores.csv")
    )

    scores = pd.concat(
        [qwen_scores, gemma_scores],
        ignore_index=True
    )

    return data, scores


data, scores = load_data()

st.title("LLM Stereotype Bias Analysis")
st.caption(
    "Comparing Baseline and Debiased prompting "
    "using the BBQ Bias Benchmark"
)

st.divider()
st.subheader("Select Evaluation")

c1, c2, c3 = st.columns(3)

with c1:
    model = st.selectbox("Model", ["Qwen", "Gemma"])

with c2:
    category = st.selectbox(
        "Category",
        ["Gender_identity", "Religion", "Race_ethnicity"],
        format_func=lambda x: x.replace("_", " ")
    )

with c3:
    context = st.selectbox(
        "Context Type",
        ["ambig", "disambig"],
        format_func=lambda x:
            "Ambiguous" if x == "ambig" else "Disambiguated"
    )

# Prepare datasets
keys = ["category", "context_condition", "question"]

baseline = data[f"{model} Baseline"].drop_duplicates(
    subset=keys, keep="first"
).copy()

debiased = data[f"{model} Debiased"].drop_duplicates(
    subset=keys, keep="first"
).copy()

# Match the same questions
comparison = pd.merge(
    baseline,
    debiased,
    on=keys,
    suffixes=("_baseline", "_debiased"),
    validate="one_to_one"
)

# Keep only selected category and context
comparison = comparison[
    (comparison["category"] == category) &
    (comparison["context_condition"] == context)
].copy()

# Keep only questions where answers differ
comparison["different"] = (
    comparison["predicted_label_baseline"].astype(int) !=
    comparison["predicted_label_debiased"].astype(int)
)

comparison = comparison[
    comparison["different"]
].reset_index(drop=True)

st.divider()
st.subheader("Baseline vs Debiased Comparison")

if comparison.empty:
    st.warning(
        "No questions with different answers were found "
        "for this selection. Try another category or context."
    )
    st.stop()

st.success(
    f"{len(comparison)} questions found where "
    "Baseline and Debiased answers differ."
)

# Select question
question = st.selectbox(
    "Select Question",
    comparison["question"].tolist()
)

row = comparison[
    comparison["question"] == question
].iloc[0]

# Retrieve original BBQ context and answer choices
reference = data["Qwen Baseline"]

ref_rows = reference[
    (reference["category"] == category) &
    (reference["context_condition"] == context) &
    (reference["question"] == question)
]

options = {}
context_text = "Context unavailable"

if not ref_rows.empty:
    ref = ref_rows.iloc[0]

    if pd.notna(ref.get("context")):
        context_text = str(ref["context"])

    for i in range(3):
        value = ref.get(f"ans{i}", None)
        if pd.notna(value):
            options[i] = str(value)

# Get selected and correct answers
def answer_text(label):
    if pd.isna(label):
        return "Unavailable"

    label = int(label)

    return options.get(label, f"Answer choice {label + 1}")


base_answer = answer_text(row["predicted_label_baseline"])
debiased_answer = answer_text(row["predicted_label_debiased"])
correct_answer = answer_text(row["answer_label_baseline"])

base_correct = (
    int(row["predicted_label_baseline"]) ==
    int(row["answer_label_baseline"])
)

debiased_correct = (
    int(row["predicted_label_debiased"]) ==
    int(row["answer_label_debiased"])
)

# Display question and context
st.markdown("### Question")
st.write(question)

st.markdown("### Context")
st.write(context_text)

st.markdown("### Answer Options")

if options:
    for i, option in sorted(options.items()):
        st.write(f"**{i}.** {option}")
else:
    st.info("Answer options are unavailable.")

st.divider()

# Side-by-side comparison
left, right = st.columns(2)

with left:
    st.markdown("### Baseline")
    st.info(base_answer)

    if base_correct:
        st.success("Correct")
    else:
        st.error("Incorrect")

with right:
    st.markdown("### Debiased")
    st.info(debiased_answer)

    if debiased_correct:
        st.success("Correct")
    else:
        st.error("Incorrect")

st.markdown(f"**Correct Answer:** {correct_answer}")

# Overall metrics
st.divider()
st.subheader("Overall Results Comparison")

def get_score(model_name, prompt, condition):
    rows = scores[
        scores["model"].astype(str).str.startswith(
            model_name, na=False
        )
        & scores["model"].astype(str).str.contains(
            prompt, case=False, na=False
        )
        & (scores["condition"] == condition)
    ]

    return rows.iloc[0] if not rows.empty else None


base_score = get_score(model, "Baseline", context)
debiased_score = get_score(model, "Debiased", context)

if base_score is not None and debiased_score is not None:

    left, right = st.columns(2)

    with left:
        st.markdown("### Baseline")
        st.metric(
            "Accuracy",
            f"{base_score['accuracy'] * 100:.2f}%"
        )
        st.metric(
            "Stereotype Answers",
            int(base_score["stereotype_answers"])
        )
        st.metric(
            "BBQ Bias Score",
            f"{base_score['BBQ_bias_score']:.4f}"
        )

    with right:
        st.markdown("### Debiased")
        st.metric(
            "Accuracy",
            f"{debiased_score['accuracy'] * 100:.2f}%",
            delta=f"{(debiased_score['accuracy'] - base_score['accuracy']) * 100:+.2f} percentage points"
        )
        st.metric(
            "Stereotype Answers",
            int(debiased_score["stereotype_answers"]),
            delta=int(
                debiased_score["stereotype_answers"] -
                base_score["stereotype_answers"]
            ),
            delta_color="inverse"
        )
        st.metric(
            "BBQ Bias Score",
            f"{debiased_score['BBQ_bias_score']:.4f}"
        )

st.divider()
st.caption(
    "Dataset: BBQ | Categories: Gender Identity, Religion, "
    "Race & Ethnicity"
)
