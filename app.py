import streamlit as st

from src.rag import (
    build_knowledge_base,
    ask_rag
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Real Estate RAG Assistant",
    page_icon="🏠",
    layout="wide"
)


st.title(
    "🏠 Real Estate Research Assistant"
)

st.caption(
    "Multi-source RAG with semantic retrieval, "
    "reranking and grounded citations."
)


# ============================================================
# SESSION STATE
# ============================================================

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None


# ============================================================
# SIDEBAR — DATA INGESTION
# ============================================================

st.sidebar.header(
    "📚 Knowledge Base"
)


urls_text = st.sidebar.text_area(
    "Enter URLs — one per line",
    height=220,
    placeholder=(
        "https://example.com/article-1\n"
        "https://example.com/article-2"
    )
)


process_button = st.sidebar.button(
    "Process Sources",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS URLS
# ============================================================

if process_button:

    urls = [
        url.strip()
        for url in urls_text.splitlines()
        if url.strip()
    ]

    if not urls:

        st.sidebar.warning(
            "Enter at least one URL."
        )

    else:

        progress = st.sidebar.progress(
            0
        )

        status_text = st.sidebar.empty()


        def update_progress(
            current,
            total,
            url,
            success,
            message
        ):

            progress.progress(
                current / total
            )

            if success:

                status_text.caption(
                    f"✅ Loaded {current}/{total}"
                )

            else:

                status_text.caption(
                    f"⚠️ Skipped {current}/{total}"
                )


        try:

            with st.spinner(
                "Loading, cleaning, chunking "
                "and embedding sources..."
            ):

                knowledge_base = (
                    build_knowledge_base(
                        urls,
                        progress_callback=update_progress
                    )
                )

            st.session_state.knowledge_base = (
                knowledge_base
            )

            progress.progress(1.0)

            status_text.success(
                "Knowledge base ready."
            )

        except Exception as exc:

            st.error(
                f"Failed to build knowledge base: {exc}"
            )


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

kb = st.session_state.knowledge_base


if kb is None:

    st.info(
        "Add source URLs in the sidebar "
        "and click **Process Sources**."
    )

    st.stop()


documents = kb["documents"]
chunks = kb["chunks"]
domains = kb["domains"]
failures = kb["failures"]


col1, col2, col3 = st.columns(3)

col1.metric(
    "Sources",
    len(documents)
)

col2.metric(
    "Chunks",
    len(chunks)
)

col3.metric(
    "Domains",
    len(domains)
)


# ============================================================
# FAILED URLS
# ============================================================

if failures:

    with st.expander(
        f"⚠️ {len(failures)} source(s) could not be loaded"
    ):

        for failure in failures:

            st.write(
                failure["url"]
            )

            st.caption(
                failure["error"]
            )


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

with st.sidebar.expander(
    "⚙️ Retrieval Settings",
    expanded=False
):

    selected_domains = st.multiselect(
        "Search domains",
        options=domains,
        default=domains
    )

    use_multi_query = st.checkbox(
        "Multi-query retrieval",
        value=True
    )

    retrieval_method = st.selectbox(
        "Retrieval method",
        [
            "Similarity",
            "MMR"
        ]
    )

    final_k = st.slider(
        "Final context chunks",
        min_value=3,
        max_value=8,
        value=5
    )

    max_per_source = st.slider(
        "Max chunks per source",
        min_value=1,
        max_value=4,
        value=2
    )


# ============================================================
# QUESTION
# ============================================================

st.markdown("---")

st.subheader(
    "💬 Ask the Research Assistant"
)


question = st.text_input(
    "Question",
    placeholder=(
        "What were the major trends in "
        "India's real estate market in Q3 2026?"
    )
)


ask_button = st.button(
    "Ask",
    type="primary"
)


# ============================================================
# RAG QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Enter a question."
        )

    elif not selected_domains:

        st.warning(
            "Select at least one domain."
        )

    else:

        try:

            with st.spinner(
                "Retrieving and analysing sources..."
            ):

                result = ask_rag(
                    vectorstore=kb[
                        "vectorstore"
                    ],

                    question=question,

                    allowed_domains=(
                        selected_domains
                    ),

                    use_multi_query=(
                        use_multi_query
                    ),

                    use_mmr=(
                        retrieval_method
                        == "MMR"
                    ),

                    final_k=final_k,

                    max_per_source=(
                        max_per_source
                    )
                )

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.subheader(
                "📌 Answer"
            )

            st.markdown(
                result["answer"]
            )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            if result["sources"]:

                st.subheader(
                    "🔗 Sources"
                )

                for source in result[
                    "sources"
                ]:

                    st.markdown(
                        f"**[Source "
                        f"{source['number']}]** "
                        f"[{source['title']}]"
                        f"({source['url']})"
                    )

                    st.caption(
                        source["domain"]
                    )


            # ------------------------------------------------
            # DEBUG
            # ------------------------------------------------

            with st.expander(
                "🔍 Retrieval Debug"
            ):

                st.markdown(
                    "#### Search Queries"
                )

                for query in (
                    result["debug"][
                        "queries"
                    ]
                ):

                    st.write(
                        f"• {query}"
                    )


                st.markdown(
                    "#### Candidate Chunks"
                )

                st.write(
                    result["debug"][
                        "candidate_count"
                    ]
                )


                st.markdown(
                    "#### Final Reranked Chunks"
                )

                for index, chunk in enumerate(
                    result["debug"][
                        "final_chunks"
                    ],
                    start=1
                ):

                    st.markdown(
                        f"**Chunk {index} — "
                        f"{chunk['domain']}**"
                    )

                    st.caption(
                        f"Rerank score: "
                        f"{chunk['rerank_score']}"
                    )

                    st.write(
                        chunk["preview"]
                    )

        except Exception as exc:

            st.error(
                f"Error generating answer: {exc}"
            )