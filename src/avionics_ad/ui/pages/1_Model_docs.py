"""Model reference: what each detector does, its parameters, and what it outputs.

Parameter tables are read from the model registry, so they cannot drift from
the code. The prose is hand-written and kept short.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st  # noqa: E402

import avionics_ad.models as M  # noqa: E402
from avionics_ad.ui import logic as L  # noqa: E402

st.set_page_config(page_title="Model docs", page_icon=":material/menu_book:", layout="wide")

st.title("Model docs")
st.caption("What each detector does, the parameters it takes, and what it outputs.")

# ---------------------------------------------------------------------------
# shared pipeline
# ---------------------------------------------------------------------------

st.header("Common to every model")
st.markdown(
    """
| Step | What happens |
|---|---|
| **Input** | Telemetry is min-max scaled (fitted on clean data only) and cut into windows of shape *window × channels*, every *stride* samples. |
| **Fit** | The model is trained on windows from the **clean** run only. It never sees an anomaly. |
| **Score** | Each test window gets one **anomaly score**. Higher means stranger. The units differ per model, so scores are not comparable across models. |
| **Threshold** | A cut-off is derived from the scores on the *training* windows (rule chosen in the sidebar). Test windows scoring above it are **flagged**. |
| **Evaluate** | Flags are compared with the known injected faults. |
"""
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Outputs")
    st.markdown(
        """
| Output | Meaning |
|---|---|
| Recall | Share of anomalous windows flagged |
| Precision | Share of flags that were real |
| F2 | Recall weighted 4× over precision |
| False negative rate | Share of anomalies missed (1 − recall) |
| Per-kind recall | Recall for level shift, pattern, and time shift separately |
| Best achievable F2 | Best F2 any threshold could reach on these scores |
| Attribution | Channels with the most error in flagged windows (deep models only) |
| MSE / MAE | How well the model reproduces normal data (deep models only) |
"""
    )
with c2:
    st.subheader("Threshold rules")
    st.dataframe(
        [{"rule": r, "how": d} for r, d in L.RULE_DESCRIPTIONS.items()],
        hide_index=True, width="stretch",
    )

st.subheader("Training parameters shared by the deep models")
st.caption(
    f"Used by every model except LOF and One-Class SVM. The app lowers the default "
    f"`epochs` to {L.UI_EPOCH_DEFAULT} so runs finish quickly."
)
shared = L.param_table("lstm")
st.dataframe(shared[shared["parameter"].isin(L.SHARED_DEEP_PARAMS)], hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# per model
# ---------------------------------------------------------------------------

#: name -> (how it works, what the score is). Hand-written; keep each to a line or two.
DOCS: dict[str, tuple[str, str]] = {
    "lof": (
        "Turns each window into features (*summary*: mean, std, min, max, net change and mean "
        "step size per channel; *flatten*: the raw values), then compares its local density with "
        "that of its *n_neighbors* nearest training windows. Handles normal data with several modes.",
        "Negative LOF score. About 1 for a typical window, larger in sparse regions.",
    ),
    "ocsvm": (
        "Builds the same window features as LOF, standardises them, and fits one RBF-kernel "
        "boundary around normal data. *nu* is the share of training windows allowed outside it. "
        "Best when normal operation is a single coherent mode.",
        "Negative SVM decision score: distance outside the boundary. Can be negative, so the "
        "log scale is unavailable.",
    ),
    "lstm": (
        "Reads the first *window − horizon* steps one at a time and predicts the last *horizon* "
        "steps. Baseline for the hybrid models.",
        "Mean squared error between the predicted and actual final steps.",
    ),
    "cnn_lstm": (
        "1-D convolutions extract local cross-channel patterns, then an LSTM models how they "
        "evolve and predicts the next steps. Aimed at broken relationships between channels.",
        "Mean squared prediction error, as for the LSTM.",
    ),
    "cnn_gru": (
        "Same as CNN+LSTM with GRU cells: about a quarter fewer recurrent parameters, faster "
        "to train.",
        "Mean squared prediction error, as for the LSTM.",
    ),
    "lstm_ae": (
        "An LSTM encoder compresses the window into one vector and an LSTM decoder rebuilds the "
        "whole window from it. Only patterns seen in training survive. Weaker on time shifts.",
        "Mean squared reconstruction error over the window.",
    ),
    "cnn_vae": (
        "A convolutional encoder maps the window to a latent distribution. A KL term "
        "(*kl_weight*) keeps it near a standard normal, so the model can't simply memorise its input.",
        "Mean squared reconstruction error, using the latent mean. With *score_includes_kl* on, "
        "it adds *kl_weight* × KL divergence.",
    ),
    "tfa_cnn_ae_hdbscan": (
        "Each window becomes a spectrogram (STFT, *nperseg*). A conv autoencoder embeds it, and "
        "HDBSCAN finds the modes of normal operation among the training embeddings.",
        "Distance to the nearest cluster centre, divided by the median training distance (about 1 "
        "is normal). *blend_reconstruction* mixes in reconstruction error. If no clusters are "
        "found, it falls back to reconstruction error and the app says so.",
    ),
}

FAMILY_LABEL = {
    "classical": "Classical: density / boundary on window features",
    "forecast": "Forecasting: predict the next steps, score the miss",
    "reconstruct": "Reconstruction: rebuild the window, score the error",
    "cluster": "Clustering: distance from the learned modes of normal",
}

st.header("Models")
by_family: dict[str, list] = {}
for info in M.available():
    by_family.setdefault(info.family, []).append(info)

for family, infos in by_family.items():
    st.subheader(FAMILY_LABEL.get(family, family))
    for info in infos:
        how, output = DOCS.get(info.name, ("", ""))
        with st.expander(f"**{info.label}**  ·  `{info.name}`"):
            st.markdown(f"**How it works:** {how or info.description}")
            st.markdown(f"**Score:** {output or 'Higher means more anomalous.'}")
            params = L.model_specific_params(info.name)
            if len(params):
                deep = any(p.name == "epochs" for p in info.params_spec)
                st.markdown("**Parameters**" + (" (plus the shared training parameters above)" if deep else ""))
                st.dataframe(params, hide_index=True, width="stretch")
            with st.popover("Full description"):
                st.markdown(info.description)
