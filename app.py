import streamlit as st

import pandas as pd

import altair as alt

import numpy as np

from pathlib import Path

import torch

import open_clip

from PIL import Image, ImageFilter, ImageDraw



#page config

st.set_page_config(

    page_title="Zero-Shot Image Retrieval Robustness",

    page_icon="🔍",

    layout="wide",

)


#model

@st.cache_resource(max_entries=1)

def load_clip_model(model_choice):

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if model_choice == "CLIP ViT-B/32":

        model_name = "ViT-B-32"

        pretrained = "openai"

        model, _, preprocess = open_clip.create_model_and_transforms(

            model_name,

            pretrained=pretrained,

        )

        tokenizer = open_clip.get_tokenizer(model_name)

    elif model_choice == "CLIP ViT-L/14":

        model_name = "ViT-L-14"

        pretrained = "openai"

        model, _, preprocess = open_clip.create_model_and_transforms(

            model_name,

            pretrained=pretrained,

        )

        tokenizer = open_clip.get_tokenizer(model_name)

    elif model_choice == "SigLIP ViT-B/16":

        model_name = "hf-hub:timm/ViT-B-16-SigLIP"

        model, _, preprocess = open_clip.create_model_and_transforms(

            model_name

        )

        tokenizer = open_clip.get_tokenizer(

            model_name

        )

    else:

        raise ValueError(

            f"Unknown model: {model_choice}"

        )

    model = model.to(device)

    model.eval()

    return model, preprocess, tokenizer, device



#degradation parameters

SEVERITY_PARAMS = {

    "mild": {

        "blur": 1,

        "noise": 0.05,

        "occlusion": 0.10,

    },

    "medium": {

        "blur": 2,

        "noise": 0.10,

        "occlusion": 0.25,

    },

    "severe": {

        "blur": 4,

        "noise": 0.20,

        "occlusion": 0.40,

    },

}



#image degradations

def apply_blur(image, severity):

    radius = SEVERITY_PARAMS[severity]["blur"]

    return image.filter(

        ImageFilter.GaussianBlur(radius=radius)

    )



def apply_noise(image, severity, seed=42):

    std = SEVERITY_PARAMS[severity]["noise"]

    array = np.asarray(image).astype(np.float32) / 255.0

    rng = np.random.default_rng(seed)

    noise = rng.normal(

        loc=0.0,

        scale=std,

        size=array.shape,

    )

    noisy = np.clip(array + noise, 0.0, 1.0)

    noisy = (noisy * 255.0).astype(np.uint8)

    return Image.fromarray(noisy)



def apply_occlusion(image, severity):

    fraction = SEVERITY_PARAMS[severity]["occlusion"]

    image = image.copy()

    width, height = image.size

    side = int(np.sqrt(fraction * width * height))

    side = min(side, width, height)

    left = (width - side) // 2

    top = (height - side) // 2

    right = left + side

    bottom = top + side

    draw = ImageDraw.Draw(image)

    draw.rectangle(

        [left, top, right, bottom],

        fill=(0, 0, 0),

    )

    return image



def apply_degradation(

    image,

    degradation,

    severity,

    seed=42,

):

    if degradation == "Clean":

        return image.copy()

    output = image.copy()

    if "Blur" in degradation:

        output = apply_blur(

            output,

            severity,

        )

    if "Noise" in degradation:

        output = apply_noise(

            output,

            severity,

            seed=seed,

        )

    if "Occlusion" in degradation:

        output = apply_occlusion(

            output,

            severity,

        )

    return output



#gallery encoding

@st.cache_data(show_spinner=False)
def encode_gallery(
    model_choice,
    degradation="Clean",
    severity="mild",
):

    model_filenames = {
        "CLIP ViT-B/32": "clip_vit_b32",
        "CLIP ViT-L/14": "clip_vit_l14",
        "SigLIP ViT-B/16": "siglip_vit_b16",
    }

    if model_choice not in model_filenames:
        raise ValueError(
            f"Unknown model: {model_choice}"
        )

    model_slug = model_filenames[model_choice]

    if degradation == "Clean":
        condition_slug = "clean"
    else:
        degradation_slug = (
            degradation
            .lower()
            .replace(" + ", "_")
            .replace(" ", "_")
        )

        condition_slug = (
            f"{degradation_slug}_{severity}"
        )

    embedding_path = Path(
        "embeddings"
    ) / (
        f"{model_slug}__{condition_slug}.npz"
    )

    if not embedding_path.exists():
        raise RuntimeError(
            "Precomputed gallery embeddings were not found: "
            f"{embedding_path}"
        )

    with np.load(
        embedding_path,
        allow_pickle=False,
    ) as data:
        image_features = data[
            "features"
        ].astype(
            np.float32
        )

        image_names = data[
            "image_names"
        ].astype(str)

    image_paths = [
        Path("images") / image_name
        for image_name in image_names
    ]

    if len(image_paths) != len(image_features):
        raise RuntimeError(
            "Embedding/image count mismatch in "
            f"{embedding_path}"
        )

    missing_images = [
        str(path)
        for path in image_paths
        if not path.exists()
    ]

    if missing_images:
        raise RuntimeError(
            "Some gallery images referenced by the embeddings "
            "are missing from images/."
        )

    return (
        image_paths,
        torch.from_numpy(image_features),
    )



#result data

@st.cache_data

def load_results():

    master_results = pd.read_csv(

        "data/flickr8k_master_results.csv"

    )

    robustness_results = pd.read_csv(

        "data/flickr8k_robustness_analysis.csv"

    )

    successive_results = pd.read_csv(

        "data/flickr8k_successive_degradation_results.csv"

    )

    mitigation_results = pd.read_csv(

        "data/flickr8k_mitigation_comparison.csv"

    )

    combined_results = pd.read_csv(

        "data/paper_table_severe_combined.csv"

    )

    return (

        master_results,

        robustness_results,

        successive_results,

        mitigation_results,

        combined_results,

    )



(

    master_results,

    robustness_results,

    successive_results,

    mitigation_results,

    combined_results,

) = load_results()



#styling

st.markdown(
    """
<style>
:root {
    --bg: #08101f;
    --panel: rgba(15, 23, 42, 0.78);
    --panel-strong: rgba(15, 23, 42, 0.94);
    --border: rgba(148, 163, 184, 0.20);
    --text: #f8fafc;
    --muted: #94a3b8;
    --blue: #60a5fa;
    --purple: #a78bfa;
}

.stApp {
    background:
        radial-gradient(circle at 78% -5%, rgba(59, 130, 246, 0.13), transparent 30%),
        radial-gradient(circle at 16% 18%, rgba(124, 58, 237, 0.08), transparent 26%),
        linear-gradient(180deg, #08101f 0%, #0b1220 48%, #0f172a 100%);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 3.2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.025em;
}

p {
    line-height: 1.65;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input {
    background: rgba(15, 23, 42, 0.88);
    color: var(--text);
    border: 1px solid rgba(148, 163, 184, 0.24);
    border-radius: 14px;
    min-height: 50px;
    transition: border-color .18s ease, box-shadow .18s ease;
}

div[data-baseweb="select"] > div:hover,
div[data-testid="stTextInput"] input:hover {
    border-color: rgba(96, 165, 250, 0.55);
}

div[data-testid="stTextInput"] input:focus {
    border-color: rgba(96, 165, 250, 0.75);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.10);
}

div.stButton > button {
    width: 100%;
    min-height: 52px;
    background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
    color: white;
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 14px;
    font-weight: 700;
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.20);
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.06);
    box-shadow: 0 13px 34px rgba(37, 99, 235, 0.30);
}

div[data-testid="stMetric"] {
    min-height: 138px;
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.72));
    border: 1px solid rgba(96, 165, 250, 0.22);
    padding: 1rem 1.15rem;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #cbd5e1;
    font-weight: 650;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: clamp(1.55rem, 2.5vw, 2.35rem);
    line-height: 1.15;
}

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.16);
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 14px;
    overflow: hidden;
}

hr {
    border-color: rgba(148, 163, 184, 0.14) !important;
}

[data-testid="stDecoration"] {
    display: none;
}

.project-hero {
    position: relative;
    overflow: hidden;
    padding: 1.65rem 1.75rem 1.75rem;
    margin: .25rem 0 1.45rem;
    border: 1px solid rgba(96, 165, 250, 0.18);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(17, 24, 39, 0.76));
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.18);
}

.project-hero::after {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    right: -150px;
    top: -230px;
    border-radius: 50%;
    background: rgba(59, 130, 246, .16);
    filter: blur(8px);
}

.project-badge {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .38rem .78rem;
    border: 1px solid rgba(96, 165, 250, .34);
    border-radius: 999px;
    color: #93c5fd;
    background: rgba(15, 23, 42, .64);
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .08em;
}

.project-title {
    position: relative;
    z-index: 1;
    margin: 1rem 0 .65rem;
    font-size: clamp(2.25rem, 4.8vw, 4.1rem);
    line-height: 1.02;
    font-weight: 800;
    letter-spacing: -.045em;
    background: linear-gradient(90deg, #f8fafc 0%, #bfdbfe 48%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.project-subtitle {
    position: relative;
    z-index: 1;
    max-width: 980px;
    margin: 0;
    color: #9aa9bf;
    font-size: 1.05rem;
    line-height: 1.65;
}

.hero-chips {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: .55rem;
    margin-top: 1.15rem;
}

.hero-chip {
    padding: .34rem .65rem;
    border-radius: 999px;
    background: rgba(30, 41, 59, .72);
    border: 1px solid rgba(148, 163, 184, .17);
    color: #cbd5e1;
    font-size: .76rem;
}
</style>
""",
    unsafe_allow_html=True,
)


#page top

st.markdown(
    """
<div class="project-hero">
    <div class="project-badge">ZERO-SHOT RETRIEVAL &nbsp;•&nbsp; ROBUSTNESS STUDY</div>
    <div class="project-title">Zero-Shot Image Retrieval<br>Under Visual Degradations</div>
    <p class="project-subtitle">
        Interactive evaluation of CLIP and SigLIP robustness under blur,
        Gaussian noise, partial occlusion, and compounded visual corruptions.
    </p>
    <div class="hero-chips">
        <span class="hero-chip">CLIP ViT-B/32</span>
        <span class="hero-chip">CLIP ViT-L/14</span>
        <span class="hero-chip">SigLIP ViT-B/16</span>
        <span class="hero-chip">Flickr8k</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


#tabs

tab1, tab2, tab3, tab4 = st.tabs(

    [

        "Interactive Retrieval",

        "Robustness Results",

        "Successive Degradation",

        "Mitigation Experiment",

    ]

)



#tab1- interactive retrieval

with tab1:

    st.header("Interactive Retrieval")

    st.write(

        "Search the 300-image Flickr8k demo gallery using any "

        "natural-language query and compare clean retrieval against "

        "retrieval under visual corruption."

    )

    #controls

    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:

        model_choice = st.selectbox(

            "Model",

            [

                "CLIP ViT-B/32",

                "CLIP ViT-L/14",

                "SigLIP ViT-B/16",

            ],

        )

    with control_col2:

        degradation_choice = st.selectbox(

            "Degradation",

            [

                "Clean",

                "Blur",

                "Noise",

                "Occlusion",

                "Blur + Noise",

                "Blur + Occlusion",

                "Noise + Occlusion",

                "Blur + Noise + Occlusion",

            ],

        )

    with control_col3:

        severity_choice = st.selectbox(

            "Severity",

            [

                "Mild",

                "Medium",

                "Severe",

            ],

            disabled=(degradation_choice == "Clean"),

        )

    query = st.text_input(

        "Text Query",

        placeholder="e.g. a dog running through grass",

    )

    search_clicked = st.button(

        "Run Retrieval",

        use_container_width=True,

    )

    #condition description

    if degradation_choice != "Clean":

        severity_key = severity_choice.lower()

        params = SEVERITY_PARAMS[severity_key]

        parameter_parts = []

        if "Blur" in degradation_choice:

            parameter_parts.append(

                f"blur radius = {params['blur']}"

            )

        if "Noise" in degradation_choice:

            parameter_parts.append(

                f"noise σ = {params['noise']}"

            )

        if "Occlusion" in degradation_choice:

            parameter_parts.append(

                f"occlusion = {int(params['occlusion'] * 100)}%"

            )

        st.caption(

            "Applied parameters: "

            + " • ".join(parameter_parts)

        )

    else:

        severity_key = "mild"

        st.caption(

            "Clean gallery — no visual corruption applied."

        )

    #run retrieval

    if search_clicked:

        if not query.strip():

            st.warning(

                "Enter a text query before running retrieval."

            )

        else:

            with st.spinner(

                "Running clean and degraded retrieval..."

            ):

                #load selected model

                (

                    model,

                    _,

                    tokenizer,

                    device,

                ) = load_clip_model(

                    model_choice

                )

                #clean gallery

                (

                    clean_image_paths,

                    clean_image_features,

                ) = encode_gallery(

                    model_choice=model_choice,

                    degradation="Clean",

                    severity="mild",

                )

                #degraded gallery

                if degradation_choice == "Clean":

                    degraded_image_paths = clean_image_paths

                    degraded_image_features = clean_image_features

                else:

                    (

                        degraded_image_paths,

                        degraded_image_features,

                    ) = encode_gallery(

                        model_choice=model_choice,

                        degradation=degradation_choice,

                        severity=severity_key,

                    )

                #text embedding

                text = tokenizer(

                    [query]

                ).to(device)

                with torch.no_grad():

                    text_features = model.encode_text(

                        text

                    )

                    text_features = (

                        text_features

                        / text_features.norm(

                            dim=-1,

                            keepdim=True,

                        )

                    )

                text_features_cpu = (

                    text_features.cpu()

                )

                #clean similarities

                clean_similarities = (

                    text_features_cpu

                    @ clean_image_features.T

                )

                (

                    clean_scores,

                    clean_indices,

                ) = clean_similarities[0].topk(5)

                #degraded similarities

                degraded_similarities = (

                    text_features_cpu

                    @ degraded_image_features.T

                )

                (

                    degraded_scores,

                    degraded_indices,

                ) = degraded_similarities[0].topk(5)

            #results

            st.divider()

            st.subheader(

                "Clean vs Degraded Retrieval"

            )

            st.caption(

                f'Query: "{query}"  •  Model: {model_choice}'

            )

            #summary metrics

            clean_top1_index = (

                clean_indices[0].item()

            )

            degraded_top1_index = (

                degraded_indices[0].item()

            )

            same_top1 = (

                clean_top1_index

                == degraded_top1_index

            )

            clean_top5_set = set(

                index.item()

                for index in clean_indices

            )

            degraded_top5_set = set(

                index.item()

                for index in degraded_indices

            )

            overlap_count = len(

                clean_top5_set

                & degraded_top5_set

            )

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.metric(

                    "Top-1 Changed",

                    "No" if same_top1 else "Yes",

                )

            with metric2:

                st.metric(

                    "Top-5 Overlap",

                    f"{overlap_count} / 5",

                )
            with metric3:
                condition_short = {
        	    "Clean": "Clean",
        	    "Blur": "Blur",
        	    "Noise": "Noise",
        	    "Occlusion": "Occlusion",
        	    "Blur + Noise": "Blur + Noise",
        	    "Blur + Occlusion": "Blur + Occlusion",
        	    "Noise + Occlusion": "Noise + Occlusion",
        	    "Blur + Noise + Occlusion": "Triple",
    		}[degradation_choice]

                st.metric(
                    "Condition",
                    "Clean"
                    if degradation_choice == "Clean"
                    else f"{condition_short} · {severity_choice}",
                )

            st.write("")

            #side-by-side panels

            left_col, right_col = st.columns(

                2,

                gap="large",

            )

            #clean side

            with left_col:

                st.markdown(

                    "### Clean Gallery"

                )

                st.caption(

                    "Retrieval using the original, uncorrupted images."

                )

                # Top result displayed large

                clean_top_index = (

                    clean_indices[0].item()

                )

                clean_top_path = (

                    clean_image_paths[

                        clean_top_index

                    ]

                )

                clean_top_image = Image.open(

                    clean_top_path

                ).convert("RGB")

                st.image(

                    clean_top_image,

                    use_container_width=True,

                )

                clean_name_col, clean_score_col = (

                    st.columns([2, 1])

                )

                with clean_name_col:

                    st.markdown(

                        "**#1 Clean Result**"

                    )

                with clean_score_col:

                    st.markdown(

                        f"**Similarity: "

                        f"{clean_scores[0].item():.3f}**"

                    )

                #remaining four results

                clean_small_cols = st.columns(4)

                for position in range(1, 5):

                    gallery_index = (

                        clean_indices[

                            position

                        ].item()

                    )

                    image_path = (

                        clean_image_paths[

                            gallery_index

                        ]

                    )

                    image = Image.open(

                        image_path

                    ).convert("RGB")

                    with clean_small_cols[

                        position - 1

                    ]:

                        st.image(

                            image,

                            use_container_width=True,

                        )

                        st.markdown(

                            f"**#{position + 1}**"

                        )

                        st.caption(

                            f"{clean_scores[position].item():.3f}"

                        )

            #degraded side

            with right_col:

                if degradation_choice == "Clean":

                    degraded_title = (

                        "Clean Gallery"

                    )

                else:

                    degraded_title = (

                        f"{degradation_choice} "

                        f"• {severity_choice}"

                    )

                st.markdown(

                    f"### {degraded_title}"

                )

                if degradation_choice == "Clean":

                    st.caption(

                        "No degradation is applied."

                    )

                else:

                    st.caption(

                        "The same gallery after the selected "

                        "visual corruption is applied."

                    )

                #top degraded result displayed large

                degraded_top_index = (

                    degraded_indices[0].item()

                )

                degraded_top_path = (

                    degraded_image_paths[

                        degraded_top_index

                    ]

                )

                degraded_top_image = Image.open(

                    degraded_top_path

                ).convert("RGB")

                degraded_top_image = apply_degradation(

                    image=degraded_top_image,

                    degradation=degradation_choice,

                    severity=severity_key,

                    seed=42 + degraded_top_index,

                )

                st.image(

                    degraded_top_image,

                    use_container_width=True,

                )

                degraded_name_col, degraded_score_col = (

                    st.columns([2, 1])

                )

                with degraded_name_col:

                    st.markdown(

                        "**#1 Degraded Result**"

                        if degradation_choice != "Clean"

                        else "**#1 Clean Result**"

                    )

                with degraded_score_col:

                    st.markdown(

                        f"**Similarity: "

                        f"{degraded_scores[0].item():.3f}**"

                    )

                #remaining four results

                degraded_small_cols = st.columns(4)

                for position in range(1, 5):

                    gallery_index = (

                        degraded_indices[

                            position

                        ].item()

                    )

                    image_path = (

                        degraded_image_paths[

                            gallery_index

                        ]

                    )

                    image = Image.open(

                        image_path

                    ).convert("RGB")

                    image = apply_degradation(

                        image=image,

                        degradation=degradation_choice,

                        severity=severity_key,

                        seed=42 + gallery_index,

                    )

                    with degraded_small_cols[

                        position - 1

                    ]:

                        st.image(

                            image,

                            use_container_width=True,

                        )

                        st.markdown(

                            f"**#{position + 1}**"

                        )

                        st.caption(

                            f"{degraded_scores[position].item():.3f}"

                        )

            #explanation

            if degradation_choice != "Clean":

                if overlap_count == 5:

                    st.success(

                        "All five retrieved images remained in the "

                        "Top-5 after degradation, although their "

                        "ranking may have changed."

                    )

                elif overlap_count >= 3:

                    st.info(

                        f"{overlap_count} of the original Top-5 "

                        "images remained after degradation. "

                        "The corruption changed part of the ranking."

                    )

                else:

                    st.warning(

                        f"Only {overlap_count} of the original Top-5 "

                        "images remained after degradation. "

                        "The selected corruption substantially changed "

                        "the retrieval ranking."

                    )



#tab2- robustness results

with tab2:

    st.header(

        "Robustness Results"

    )

    st.write(

        "Compare retrieval performance across models, "

        "degradation types, and severity levels."

    )

    model_filter = st.selectbox(

        "Select model",

        sorted(

            master_results[

                "model"

            ].unique()

        ),

        key="robustness_model",

    )

    filtered = master_results[

        master_results["model"]

        == model_filter

    ].copy()

    clean_row = filtered[

        filtered["condition"]

        == "clean"

    ].iloc[0]

    severe_rows = filtered[

        filtered["severity"]

        == "severe"

    ]

    worst_r1 = severe_rows.loc[

        severe_rows[

            "Recall@1"

        ].idxmin()

    ]

    (

        metric_col1,

        metric_col2,

        metric_col3,

        metric_col4,

    ) = st.columns(4)

    with metric_col1:

        st.metric(

            "Clean Recall@1",

            f"{clean_row['Recall@1']:.3f}",

        )

    with metric_col2:

        st.metric(

            "Clean MRR",

            f"{clean_row['MRR']:.3f}",

        )

    with metric_col3:

        st.metric(

            "Worst Severe R@1",

            f"{worst_r1['Recall@1']:.3f}",

        )

    with metric_col4:

        st.metric(

            "Worst Condition",

            worst_r1["condition"],

        )

    st.subheader(

        "Recall@1 Across Conditions"

    )

    chart_data = filtered[

        filtered["condition"]

        != "clean"

    ][

        [

            "condition",

            "severity",

            "Recall@1",

        ]

    ].copy()

    severity_order = [

        "mild",

        "medium",

        "severe",

    ]

    robustness_chart = (

        alt.Chart(chart_data)

        .mark_bar(

            cornerRadiusTopLeft=5,

            cornerRadiusTopRight=5,

        )

        .encode(

            x=alt.X(

                "condition:N",

                title="Degradation",

                axis=alt.Axis(

                    labelAngle=-30

                ),

            ),

            xOffset=alt.XOffset(

                "severity:N",

                sort=severity_order,

            ),

            y=alt.Y(

                "Recall@1:Q",

                title="Recall@1",

                scale=alt.Scale(

                    domain=[0, 0.8]

                ),

            ),

            color=alt.Color(

                "severity:N",

                title="Severity",

                sort=severity_order,

            ),

            tooltip=[

                alt.Tooltip(

                    "condition:N",

                    title="Condition",

                ),

                alt.Tooltip(

                    "severity:N",

                    title="Severity",

                ),

                alt.Tooltip(

                    "Recall@1:Q",

                    format=".4f",

                ),

            ],

        )

        .properties(

            height=430

        )

    )

    st.altair_chart(

        robustness_chart,

        use_container_width=True,

    )

    with st.expander(

        "View full results table"

    ):

        st.dataframe(

            filtered[

                [

                    "condition",

                    "severity",

                    "Recall@1",

                    "Recall@5",

                    "Recall@10",

                    "MRR",

                    "Mean Rank",

                    "Median Rank",

                ]

            ],

            use_container_width=True,

            hide_index=True,

        )



#tab3- successive degradation

with tab3:

    st.header(

        "Successive Degradation"

    )

    st.write(

        "Track how retrieval performance changes as "

        "visual corruptions accumulate."

    )

    severity_filter = st.selectbox(

        "Select severity",

        [

            "mild",

            "medium",

            "severe",

        ],

        index=2,

        key="successive_severity",

    )

    successive_filtered = (

        successive_results[

            successive_results[

                "severity"

            ]

            == severity_filter

        ].copy()

    )

    stage_order = [

        "clean",

        "blur",

        "blur+noise",

        "blur+noise+occlusion",

    ]

    st.subheader(

        "Recall@1"

    )

    r1_chart = (

        alt.Chart(

            successive_filtered

        )

        .mark_line(

            point=True,

            strokeWidth=3,

        )

        .encode(

            x=alt.X(

                "condition:N",

                title="Successive Degradation",

                sort=stage_order,

            ),

            y=alt.Y(

                "Recall@1:Q",

                title="Recall@1",

                scale=alt.Scale(

                    domain=[0, 0.8]

                ),

            ),

            color=alt.Color(

                "model:N",

                title="Model",

            ),

            tooltip=[

                "model",

                "condition",

                alt.Tooltip(

                    "Recall@1:Q",

                    format=".4f",

                ),

            ],

        )

        .properties(

            height=420

        )

        .interactive()

    )

    st.altair_chart(

        r1_chart,

        use_container_width=True,

    )

    st.subheader(

        "MRR"

    )

    mrr_chart = (

        alt.Chart(

            successive_filtered

        )

        .mark_line(

            point=True,

            strokeWidth=3,

        )

        .encode(

            x=alt.X(

                "condition:N",

                title="Successive Degradation",

                sort=stage_order,

            ),

            y=alt.Y(

                "MRR:Q",

                title="MRR",

                scale=alt.Scale(

                    domain=[0, 0.9]

                ),

            ),

            color=alt.Color(

                "model:N",

                title="Model",

            ),

            tooltip=[

                "model",

                "condition",

                alt.Tooltip(

                    "MRR:Q",

                    format=".4f",

                ),

            ],

        )

        .properties(

            height=420

        )

        .interactive()

    )

    st.altair_chart(

        mrr_chart,

        use_container_width=True,

    )

    with st.expander(

        "View successive degradation data"

    ):

        st.dataframe(

            successive_filtered,

            use_container_width=True,

            hide_index=True,

        )



#tab4- mitigation

with tab4:

    st.header(

        "Mitigation Experiment"

    )

    st.write(

        "Evaluate whether lightweight test-time image processing "

        "can recover retrieval performance under severe blur + noise."

    )

    st.info(

        "Test condition: SigLIP ViT-B/16 under severe Blur + Noise."

    )

    mitigation_display = (

        mitigation_results.copy()

    )

    baseline_row = (

        mitigation_display.iloc[0]

    )

    multiview_row = (

        mitigation_display.iloc[1]

    )

    restoration_row = (

        mitigation_display.iloc[2]

    )

    (

        method_col1,

        method_col2,

        method_col3,

    ) = st.columns(3)

    with method_col1:

        st.metric(

            "Baseline R@1",

            f"{baseline_row['Recall@1']:.4f}",

        )

        st.metric(

            "Baseline MRR",

            f"{baseline_row['MRR']:.4f}",

        )

    with method_col2:

        st.metric(

            "Multi-view R@1",

            f"{multiview_row['Recall@1']:.4f}",

            delta=(

                f"{multiview_row['Recall@1'] - baseline_row['Recall@1']:.4f}"

            ),

        )

        st.metric(

            "Multi-view MRR",

            f"{multiview_row['MRR']:.4f}",

            delta=(

                f"{multiview_row['MRR'] - baseline_row['MRR']:.4f}"

            ),

        )

    with method_col3:

        st.metric(

            "Restoration R@1",

            f"{restoration_row['Recall@1']:.4f}",

            delta=(

                f"{restoration_row['Recall@1'] - baseline_row['Recall@1']:.4f}"

            ),

        )

        st.metric(

            "Restoration MRR",

            f"{restoration_row['MRR']:.4f}",

            delta=(

                f"{restoration_row['MRR'] - baseline_row['MRR']:.4f}"

            ),

        )

    st.subheader(

        "Mitigation Comparison"

    )

    mitigation_chart_data = (

        mitigation_display.melt(

            id_vars=["method"],

            value_vars=[

                "Recall@1",

                "MRR",

            ],

            var_name="Metric",

            value_name="Score",

        )

    )

    mitigation_chart = (

        alt.Chart(

            mitigation_chart_data

        )

        .mark_bar(

            cornerRadiusTopLeft=7,

            cornerRadiusTopRight=7,

        )

        .encode(

            x=alt.X(

                "method:N",

                title="Method",

                axis=alt.Axis(

                    labelAngle=0

                ),

            ),

            xOffset=alt.XOffset(

                "Metric:N"

            ),

            y=alt.Y(

                "Score:Q",

                title="Score",

                scale=alt.Scale(

                    domain=[0, 0.30]

                ),

            ),

            color=alt.Color(

                "Metric:N",

                title="Metric",

            ),

            tooltip=[

                alt.Tooltip(

                    "method:N",

                    title="Method",

                ),

                alt.Tooltip(

                    "Metric:N",

                    title="Metric",

                ),

                alt.Tooltip(

                    "Score:Q",

                    title="Score",

                    format=".4f",

                ),

            ],

        )

        .properties(

            height=340

        )

    )

    st.altair_chart(

        mitigation_chart,

        use_container_width=True,

    )

    st.warning(

        "Neither lightweight intervention improved robustness. "

        "Multi-view averaging reduced performance slightly, while "

        "naive restoration caused a larger drop. This suggests that "

        "corruption-aware or learned adaptation may be needed rather "

        "than generic preprocessing."

    )

    with st.expander(

        "View mitigation results"

    ):

        st.dataframe(

            mitigation_display,

            use_container_width=True,

            hide_index=True,

        )
