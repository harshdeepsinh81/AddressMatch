"""
Streamlit UI for the deterministic address-matching engine.

This file is designed to sit INSIDE the address/ package folder,
alongside integration_adapter.py, datamodel.py, etc.

Two modes:
  1. Single pair -- type Address1 / Address2, see extraction + engine
     label + all four policy verdicts.
  2. CSV upload -- a CSV with columns "Address1","Address2"; every row
     is matched, results shown in a table and downloadable as CSV.

Run with (from the PARENT directory of address/, i.e. one level up
from where this file sits):
    pip install streamlit pandas
    streamlit run address/streamlit_app.py

No matching logic lives in this file -- it only calls the existing,
unmodified address.integration_adapter.match_pair() and
address.policies.evaluate_policy().
"""

import sys
import os

# This file lives INSIDE address/, but integration_adapter.py etc. use
# package-relative imports (e.g. "from .datamodel import ..."), which
# only resolve correctly when imported AS a package (address.xxx), not
# as loose sibling files. So: add the PARENT of address/ to sys.path
# (not this file's own directory), then import via the "address."
# prefix -- this works regardless of which folder this file physically
# sits in, and matches how every other module in this project is
# already imported. Verified directly: importing "from integration_adapter
# import match_pair" (treating it as a sibling) fails with
# "attempted relative import with no known parent package"; importing
# "from address.integration_adapter import match_pair" after adding the
# parent directory works correctly.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import streamlit as st
import pandas as pd

from address.integration_adapter import match_pair
from address.policies import evaluate_policy, ACTIVE_POLICY_NAMES
from address.datamodel import ComponentType


st.set_page_config(page_title="Address Matcher", layout="wide")


# ---------------------------------------------------------------------------
# Core matching call -- wraps match_pair() + all four policy evaluations
# into one plain dict, reused by both UI modes so they can't diverge.
# ---------------------------------------------------------------------------

def run_match(address1: str, address2: str) -> dict:
    result = match_pair(address1, address2)
    mr = result.match_result

    high_sim = mr.label == "HIGH"
    pin_ev = mr.evidence.get(ComponentType.PIN)
    pins_match = pin_ev.state.value == "match" if pin_ev else False

    policy_tiers = {}
    for pname in ACTIVE_POLICY_NAMES:
        decision = evaluate_policy(pname, mr, high_sim, pins_match)
        policy_tiers[pname] = decision.tier.value

    flat_ev = mr.evidence.get(ComponentType.FLAT)
    wing_ev = mr.evidence.get(ComponentType.WING)

    return {
        "address1": address1,
        "address2": address2,
        "engine_label": mr.label,
        "engine_score": round(mr.raw_score, 2),
        "unit1": result.unit_component_a.value if result.unit_component_a else None,
        "unit2": result.unit_component_b.value if result.unit_component_b else None,
        "unit_status": flat_ev.state.value if flat_ev else "missing_on_both",
        "subunit1": result.subunit_component_a.value if result.subunit_component_a else None,
        "subunit2": result.subunit_component_b.value if result.subunit_component_b else None,
        "subunit_status": wing_ev.state.value if wing_ev else "missing_on_both",
        "pin1": result.extraction_a.pin.value if result.extraction_a.pin else None,
        "pin2": result.extraction_b.pin.value if result.extraction_b.pin else None,
        "pin_status": pin_ev.state.value if pin_ev else "missing_on_both",
        "base_address1": result.extraction_a.base_address,
        "base_address2": result.extraction_b.base_address,
        **{f"policy_{p}": t for p, t in policy_tiers.items()},
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Address Matcher")
st.caption("Deterministic UNIT / SUBUNIT / PIN extraction + matching — no ML.")

mode = st.radio("Mode", ["Single pair", "Upload CSV"], horizontal=True)

if mode == "Single pair":
    col1, col2 = st.columns(2)
    with col1:
        address1 = st.text_area("Address 1", height=100, placeholder="Flat 401, Wing B, Mumbai 400053")
    with col2:
        address2 = st.text_area("Address 2", height=100, placeholder="Flat 402, Wing B, Mumbai 400053")

    if st.button("Match", type="primary"):
        if not address1.strip() or not address2.strip():
            st.warning("Enter both addresses.")
        else:
            r = run_match(address1, address2)

            label_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(r["engine_label"], "")
            st.subheader(f"{label_color} Engine result: {r['engine_label']}  (score {r['engine_score']})")

            st.markdown("**Extracted components**")
            comp_table = pd.DataFrame([
                {"Component": "UNIT/FLAT", "Address 1": r["unit1"] or "—", "Address 2": r["unit2"] or "—", "Status": r["unit_status"]},
                {"Component": "SUBUNIT/WING", "Address 1": r["subunit1"] or "—", "Address 2": r["subunit2"] or "—", "Status": r["subunit_status"]},
                {"Component": "PIN", "Address 1": r["pin1"] or "—", "Address 2": r["pin2"] or "—", "Status": r["pin_status"]},
            ])
            st.table(comp_table)

            st.markdown("**Policy verdicts**")
            policy_table = pd.DataFrame([
                {"Policy": p, "Verdict": r[f"policy_{p}"]} for p in ACTIVE_POLICY_NAMES
            ])
            st.table(policy_table)

            with st.expander("Base address (unmatched text) — not used in scoring"):
                st.text(f"A1: {r['base_address1']}")
                st.text(f"A2: {r['base_address2']}")

else:
    st.markdown("Upload a CSV with columns **Address1** and **Address2**.")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            missing_cols = {"Address1", "Address2"} - set(df.columns)
            if missing_cols:
                st.error(f"CSV is missing required column(s): {', '.join(missing_cols)}. "
                          f"Found columns: {list(df.columns)}")
            else:
                st.write(f"{len(df)} row(s) found.")
                if st.button("Run matching on all rows", type="primary"):
                    progress = st.progress(0, text="Matching...")
                    results = []
                    for i, row in df.iterrows():
                        a1 = str(row["Address1"]) if pd.notna(row["Address1"]) else ""
                        a2 = str(row["Address2"]) if pd.notna(row["Address2"]) else ""
                        if not a1.strip() or not a2.strip():
                            results.append({"address1": a1, "address2": a2, "engine_label": "SKIPPED (empty)"})
                        else:
                            results.append(run_match(a1, a2))
                        progress.progress((i + 1) / len(df), text=f"Matching... {i + 1}/{len(df)}")
                    progress.empty()

                    result_df = pd.DataFrame(results)
                    st.success(f"Done. {len(result_df)} row(s) matched.")
                    st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download results as CSV",
                        data=csv_bytes,
                        file_name="address_match_results.csv",
                        mime="text/csv",
                    )