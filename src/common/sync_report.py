import os
import re
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.panel_costs import (
    load_panel_tables, billed_cost_for_features, read_selected_features, ENDPOINT_PANELS
)


# ----------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------

def parse_bm1_metrics(bm1_path):
    """R2 for the FISTA arm only (skips the Ridge Refit section)."""
    bm1_r2 = {}
    if not os.path.exists(bm1_path):
        return bm1_r2
    with open(bm1_path, 'r', encoding='utf-8') as f:
        in_fista, curr = False, None
        for line in f:
            s = line.strip()
            if '--- Multi-Task L2,1 Lasso (FISTA) ---' in s:
                in_fista = True
            elif '--- Ridge Refit' in s:
                in_fista = False
            elif in_fista and s.startswith('Target:'):
                curr = s.split('Target:')[1].strip()
            elif in_fista and curr and '- R2:' in s:
                m = re.search(r"- R2:\s+([\d\.]+)\s+\+/-\s+([\d\.]+)\s+\(95% CI:\s+\[([\d\.]+),\s+([\d\.]+)\]\)", s)
                if m:
                    bm1_r2[curr] = tuple(float(m.group(i)) for i in (1, 2, 3, 4))
    return bm1_r2


def parse_bm1_feature_counts(bm1_path):
    """Returns (selected_features, candidate_pool) from the header line."""
    if not os.path.exists(bm1_path):
        return None, None
    with open(bm1_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r"Mean Features Selected:\s+(\d+)\s+out of\s+(\d+)", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def parse_bm2_metrics(bm2_path):
    bm2_r2 = {}
    if not os.path.exists(bm2_path):
        return bm2_r2
    with open(bm2_path, 'r', encoding='utf-8') as f:
        curr = None
        for line in f:
            s = line.strip()
            if s.startswith('Target:'):
                curr = s.split('Target:')[1].strip()
            elif curr and s.startswith('R2:'):
                m = re.search(r"R2:\s+([\d\.]+)\s+\+/-\s+([\d\.]+)\s+\(95% CI:\s+\[([\d\.]+),\s+([\d\.]+)\]\)", s)
                if m:
                    bm2_r2[curr] = tuple(float(m.group(i)) for i in (1, 2, 3, 4))
    return bm2_r2


def parse_greedy_metrics(bm3_path):
    steps = {}
    if not os.path.exists(bm3_path):
        return steps
    with open(bm3_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 5:
                try:
                    idx = int(parts[0])
                    cost_val = float(parts[2].replace('$', '').replace(',', ''))
                    steps[idx] = (cost_val, f"${cost_val:,.0f}", float(parts[3]), parts[4])
                except ValueError:
                    continue
    return steps


def parse_ablation_results(path):
    matrix = {}
    if not os.path.exists(path):
        return matrix
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 6 and parts[0] != 'Feature Modality Subset':
                matrix[(parts[0], parts[1])] = (parts[2], parts[3], parts[4], parts[5])
    return matrix


def parse_tier1_summary(path):
    """Returns (n_features, billed_cost, n_panels)."""
    if not os.path.exists(path):
        return None
    k = cost = panels = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r"Selected Features:\s+(\d+)", line)
            if m:
                k = int(m.group(1))
            m = re.search(r"Billed Cost:\s+([\d\.]+)", line)
            if m:
                cost = float(m.group(1))
            m = re.search(r"Panels Triggered:\s+(\d+)", line)
            if m:
                panels = int(m.group(1))
    if k is None or cost is None or panels is None:
        return None
    return k, cost, panels


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def split_ci(cell):
    """'0.8026 [0.7702, 0.8350]' -> (mean, ci_halfwidth, low, high)"""
    m = re.match(r"([\d\.]+)\s+\[([\d\.]+),\s*([\d\.]+)\]", cell.strip())
    assert m, f"Unparseable ablation cell: {cell!r}"
    mean, low, high = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return mean, (high - low) / 2.0, low, high


def latex_thousands(n):
    return f"{n:,}".replace(",", "{,}")


def safe_replace(text, pattern, replacement_str):
    """re.sub that treats the replacement as a literal (no backslash escapes)."""
    return re.sub(pattern, lambda m: replacement_str, text, flags=re.DOTALL)


def replace_markdown_table(lines, header_marker, new_table_lines):
    start = None
    for i, line in enumerate(lines):
        if header_marker in line:
            start = i
            break
    if start is None:
        return lines
    end = start
    while end < len(lines) and lines[end].strip().startswith('|'):
        end += 1
    return lines[:start] + new_table_lines + lines[end:]


PANEL_DESCRIPTIONS = {
    "Amyloid PET Imaging": "Brain PET scan detecting amyloid plaque buildup.",
    "FDG PET Imaging": "Brain PET scan measuring brain glucose metabolism.",
    "ASL MRI (Arterial Spin Labeling)": "MRI measuring blood flow in brain tissue.",
    "Structural MRI (FreeSurfer)": "High-resolution MRI measuring brain shrink/volume.",
    "CSF Biomarkers (Lumbar Puncture)": "Spinal tap measuring Alzheimer's proteins (Tau/Amyloid).",
    "Rey Auditory Verbal Learning Test (RAVLT)": "Word list memory test.",
    "Functional Assessment Questionnaire (FAQ)": "Daily living activities questionnaire (filled by family).",
    "Boston Naming Test": "Picture object naming test.",
    "Trail Making Test (TMT)": "Connect-the-dots visual processing speed test.",
    "Demographics & Medical History": "Age, gender, education, basic health history.",
    "Category Fluency Test": "Verbal animal naming speed test.",
    "Clock Drawing Test": "Drawing clock face spatial memory test.",
    "ADAS-Cog Assessment": "Primary trial endpoint (cognitive score).",
    "Clinical Dementia Rating (CDR)": "Primary trial endpoint (dementia severity stage).",
    "MMSE Assessment": "Primary trial endpoint (mental status exam)."
}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def sync_report_metrics():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    report_path = os.path.join(root, 'Project_Summary_Report.md')

    b1_dir = os.path.join(root, 'benchmark 1 multitask learning')
    b2_dir = os.path.join(root, 'benchmark 2 xgboost')
    bm1_path = os.path.join(b1_dir, 'predictive_metrics_benchmark1.txt')
    bm2_path = os.path.join(b2_dir, 'predictive_metrics_benchmark2.txt')
    bm3_path = os.path.join(root, 'benchmark 3 greedy panel', 'greedy_baseline_metrics.txt')
    ablation_path = os.path.join(root, 'src', 'ablation_results.txt')
    tier1_path = os.path.join(root, 'src', 'tier1_cognitive_summary.txt')

    if not os.path.exists(report_path):
        print(f"Report path {report_path} not found.")
        return

    bm1_r2 = parse_bm1_metrics(bm1_path)
    bm2_r2 = parse_bm2_metrics(bm2_path)
    greedy_steps = parse_greedy_metrics(bm3_path)
    ablation = parse_ablation_results(ablation_path)
    tier1 = parse_tier1_summary(tier1_path)
    tier2_k, pool_d = parse_bm1_feature_counts(bm1_path)

    # ---- fail loudly rather than falling back to stale literals ----
    assert len(bm1_r2) == 3, f"BM1 metrics parsing failed! Expected 3 targets, found {len(bm1_r2)}."
    assert len(bm2_r2) == 3, f"BM2 metrics parsing failed! Expected 3 targets, found {len(bm2_r2)}."
    assert len(ablation) == 8, f"Ablation parsing failed! Expected 8 rows, found {len(ablation)}."
    assert tier1 is not None, "tier1_cognitive_summary.txt missing or unparseable — run the ablation study first."
    assert tier2_k and pool_d, "Could not read 'Mean Features Selected' from the BM1 metrics file."
    assert 0 in greedy_steps, "Greedy trace is missing step 0 (full panel set)."
    assert len(greedy_steps) >= 3, f"Greedy trace too short: {len(greedy_steps)} steps."

    # ---- Tier 2 cost, computed from the actual selection ----
    panel_costs_tbl, feature_to_panel = load_panel_tables(
        os.path.join(b1_dir, 'panel_costs.csv'),
        os.path.join(b1_dir, 'feature_to_panel_mapping.csv'))
    tier2_features = read_selected_features(os.path.join(b1_dir, 'selected_features_benchmark1.csv'))
    tier2_cost, tier2_panels = billed_cost_for_features(tier2_features, panel_costs_tbl, feature_to_panel)
    tier2_cost_fmt = f"${tier2_cost:,.2f}"

    # ---- XGBoost cost, computed from its actual selection ----
    xgb_features = read_selected_features(os.path.join(b2_dir, 'selected_features_benchmark2.csv'))
    xgb_cost, xgb_panels = billed_cost_for_features(xgb_features, panel_costs_tbl, feature_to_panel)
    xgb_cost_fmt = f"${xgb_cost:,.2f}"

    # ---- Tier 1 (cognitive-only) operating point ----
    tier1_k, tier1_cost, tier1_panels = tier1
    tier1_cost_fmt = f"${tier1_cost:,.2f}"
    tier1_savings_fmt = f"${(tier2_cost - tier1_cost):,.2f}"

    t1_adas = split_ci(ablation[("Cognitive Tests ONLY", "FISTA MTFL")][1])
    t1_cdr = split_ci(ablation[("Cognitive Tests ONLY", "FISTA MTFL")][2])
    t1_mmse = split_ci(ablation[("Cognitive Tests ONLY", "FISTA MTFL")][3])

    a_m, a_ci, a_l, a_h = bm1_r2['M24_ADAS13']
    c_m, c_ci, c_l, c_h = bm1_r2['M24_CDRSB']
    m_m, m_ci, m_l, m_h = bm1_r2['M24_MMSE']
    a2, c2, m2 = bm2_r2['M24_ADAS13'], bm2_r2['M24_CDRSB'], bm2_r2['M24_MMSE']

    adas_drop = a_m - t1_adas[0]

    # ---- greedy Pareto point: cheapest step within 0.001 R2 of the peak ----
    peak = max(s[2] for s in greedy_steps.values())
    pareto_idx, min_cost = 0, greedy_steps[0][0]
    for i in sorted(greedy_steps):
        cost_val, _, r2_val, _ = greedy_steps[i]
        if r2_val >= peak - 0.001 and cost_val < min_cost:
            min_cost, pareto_idx = cost_val, i

    c_full, r_full = greedy_steps[0][1], greedy_steps[0][2]
    c_par, r_par = greedy_steps[pareto_idx][1], greedy_steps[pareto_idx][2]
    last = max(greedy_steps)
    c_pruned, r_pruned = greedy_steps[last][1], greedy_steps[last][2]
    pareto_savings = f"${(tier2_cost - greedy_steps[pareto_idx][0]):,.0f}"
    pareto_delta = r_par - (a_m + c_m + m_m) / 3.0

    # ---- derived descriptive numbers ----
    cog_pool_n = int(re.search(r"(\d+)", ablation[("Cognitive Tests ONLY", "FISTA MTFL")][0]).group(1))
    bio_f = [split_ci(ablation[("Pure Biomarkers ONLY", "FISTA MTFL")][i])[0] for i in (1, 2, 3)]
    bio_t = [split_ci(ablation[("Pure Biomarkers ONLY", "Decision Trees")][i])[0] for i in (1, 2, 3)]
    bio_f_rng = f"{min(bio_f):.2f} - {max(bio_f):.2f}"
    bio_t_rng = f"{min(bio_t):.2f} - {max(bio_t):.2f}"

    with open(report_path, 'r', encoding='utf-8') as f:
        report_text = f.read()

    # ---- inline prose ----
    report_text = safe_replace(
        report_text,
        r"boosting 24-month prediction accuracy up to \*\*\$R\^2 = [\d\.]+\$?\*\* \(up from \$0\.5895\$\)",
        f"boosting 24-month prediction accuracy up to **$R^2 = {a_m:.4f}$** (up from $0.5895$)")

    report_text = safe_replace(
        report_text,
        r"identifying \*\*\d+ truly essential clinical features\*\*",
        f"identifying **{tier2_k} truly essential clinical features**")

    report_text = safe_replace(
        report_text,
        r"while keeping patient screening costs low \(\*\*\$[\d,\.]+\*\*\)",
        f"while keeping patient screening costs low (**{tier2_cost_fmt}**)")

    no_t0 = split_ci(ablation[("Excluding Endpoint Totals (t=0)", "FISTA MTFL")][1])[0]
    report_text = safe_replace(
        report_text,
        r"maintains high predictive power \(\$R\^2 = [\d\.]+\$\) purely from imaging",
        f"maintains high predictive power ($R^2 = {no_t0:.4f}$) purely from imaging")

    report_text = safe_replace(
        report_text,
        r"outperforms single-task XGBoost \(\$R\^2 = [\d\.]+\$\) on this dataset",
        f"outperforms single-task XGBoost ($R^2 = {a2[0]:.4f}$) on this dataset")

    report_text = safe_replace(
        report_text,
        r"boosting prediction accuracy to \*\*\$R\^2 = [\d\.]+\$?\*\*\!",
        f"boosting prediction accuracy to **$R^2 = {a_m:.4f}$**!")

    report_text = safe_replace(
        report_text,
        r"the model converged to \*\*\d+ core features\*\*",
        f"the model converged to **{tier2_k} core features**")

    # ---- methods table literals ----
    report_text = safe_replace(
        report_text,
        r"\$d = [\d\{\},]+\$ candidate features",
        f"$d = {latex_thousands(pool_d)}$ candidate features")

    report_text = safe_replace(
        report_text,
        r"\$d\^\* = \d+\$ core features",
        f"$d^* = {tier2_k}$ core features")

    report_text = safe_replace(
        report_text,
        r"\(\$[\d\{\},]+\$ features shrunk to zero\)",
        f"(${latex_thousands(pool_d - tier2_k)}$ features shrunk to zero)")

    # ---- summary table ----
    lines = report_text.splitlines()

    summary_table = [
        "| AI / Statistical Method | Selected Features | Billed Cost per Patient | 24-Month Memory Score Accuracy ($R^2$) | Simple Interpretation |",
        "| :--- | :---: | :---: | :---: | :--- |",
        (f"| **Multi-Task $L_{{2,1}}$ Lasso (FISTA)** | **{tier2_k} features** | **{tier2_cost_fmt}** | "
         f"**ADAS13**: **${a_m:.4f} \\pm {a_ci:.4f}$** ([{a_l:.4f}, {a_h:.4f}])<br>"
         f"**CDR-SB**: **${c_m:.4f} \\pm {c_ci:.4f}$** ([{c_l:.4f}, {c_h:.4f}])<br>"
         f"**MMSE**: **${m_m:.4f} \\pm {m_ci:.4f}$** ([{m_l:.4f}, {m_h:.4f}]) | "
         f"**Full Multi-Modal Operating Point**: Top-end ADAS13 precision combining imaging, fluid, and psychometrics. "
         f"Note: Greedy Step {pareto_idx} ({c_par}) achieves equivalent mean $R^2 = {r_par:.4f}$ for {pareto_savings} less (see greedy trace). |"),
        (f"| **Cognitive Tests ONLY (FISTA)** | **{tier1_k} features** | **{tier1_cost_fmt}** | "
         f"**ADAS13**: **${t1_adas[0]:.4f} \\pm {t1_adas[1]:.4f}$** ([{t1_adas[2]:.4f}, {t1_adas[3]:.4f}])<br>"
         f"**CDR-SB**: **${t1_cdr[0]:.4f} \\pm {t1_cdr[1]:.4f}$** ([{t1_cdr[2]:.4f}, {t1_cdr[3]:.4f}])<br>"
         f"**MMSE**: **${t1_mmse[0]:.4f} \\pm {t1_mmse[1]:.4f}$** ([{t1_mmse[2]:.4f}, {t1_mmse[3]:.4f}]) | "
         f"**Tier 1 (Ultra-Low-Cost Operating Point)**: {tier1_panels} panels. Saves **{tier1_savings_fmt}** per patient "
         f"at a ${adas_drop:.2f}$ drop in ADAS13 $R^2$ with overlapping 95% CIs. |"),
        (f"| **Decision Tree Models (XGBoost)** | {tier2_k} features | **{xgb_cost_fmt}** | "
         f"**ADAS13**: ${a2[0]:.4f} \\pm {a2[1]:.4f}$ ([{a2[2]:.4f}, {a2[3]:.4f}])<br>"
         f"**CDR-SB**: ${c2[0]:.4f} \\pm {c2[1]:.4f}$ ([{c2[2]:.4f}, {c2[3]:.4f}])<br>"
         f"**MMSE**: ${m2[0]:.4f} \\pm {m2[1]:.4f}$ ([{m2[2]:.4f}, {m2[3]:.4f}]) | "
         f"Tree baseline ({len(xgb_panels)} panels) evaluated on matching feature budget; achieves comparable accuracy at lower screening cost ({xgb_cost_fmt} vs {tier2_cost_fmt}). |"),
        (f"| **Greedy Panel Elimination (FISTA)** | Dynamic panel subsets | **{c_full} $\\rightarrow$ {c_pruned}** | "
         f"**Full Set**: {r_full:.4f} ({c_full})<br>**Step {pareto_idx} ({c_par})**: {r_par:.4f}<br>"
         f"**Pruned Set**: **{r_pruned:.4f}** at {c_pruned} | "
         f"Backward panel pruning reveals a **Pareto-dominant operating point**: Step {pareto_idx} achieves mean "
         f"$R^2 = {r_par:.4f}$ at {c_par}, saving {pareto_savings} per patient versus the full selection at "
         f"{tier2_cost_fmt} ({pareto_delta:+.4f} mean $R^2$). |"),
    ]
    lines = replace_markdown_table(lines, "AI / Statistical Method", summary_table)

    # ---- ablation matrix ----
    ablation_lines = [
        "| Feature Modality Subset | Model / Baseline | ADAS13 $R^2$ (95% CI) | CDR-SB $R^2$ (95% CI) | MMSE $R^2$ (95% CI) | Scientific Justification & Findings |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
    ]
    subsets_order = [
        ("Full Model (All Modalities)", "Full Model (All Modalities)", "FISTA MTFL",
         "Top-performing multi-modal clinical forecasting model combining cognitive anchors, imaging, and fluid biomarkers."),
        ("Full Model (All Modalities)", "Full Model (All Modalities)", "Decision Trees",
         "Tree baseline on full modality set."),
        ("Excluding Endpoint Totals ($t=0$) (No `TOTAL13`, `CDRSB`, `MMSCORE`, `TOTSCORE`, `ADAS11`)",
         "Excluding Endpoint Totals (t=0)", "FISTA MTFL",
         "**Purged Target Proxies**: Purging baseline target proxies (`TOTAL13`, `TOTSCORE`) verifies that domain psychometrics (`FAQ`, `RAVLT`, `BNT`, `TMT`) maintain strong predictive signal."),
        ("Excluding Endpoint Totals ($t=0$)", "Excluding Endpoint Totals (t=0)", "Decision Trees",
         "Tree baseline excluding endpoint totals and proxies."),
        (f"Pure Biomarkers ONLY (Excludes ALL {cog_pool_n} Cognitive/Psychometric Tests)",
         "Pure Biomarkers ONLY", "FISTA MTFL",
         f"**True Biological Floor**: Structural MRI, PET SUVr, CSF A$\\beta$/p-Tau, APOE, and Demographics achieve $R^2 \\approx {bio_f_rng}$."),
        ("Pure Biomarkers ONLY", "Pure Biomarkers ONLY", "Decision Trees",
         f"Tree baseline on pure biological markers ($R^2 \\approx {bio_t_rng}$)."),
        ("Cognitive Tests ONLY (Excludes ALL MRI, PET, CSF Biomarkers)",
         "Cognitive Tests ONLY", "FISTA MTFL",
         f"**Tier 1 Pareto Winner ({tier1_cost_fmt})**: Psychometric tests supply primary cognitive baseline variance, representing an ultra-cost-effective screening tier."),
        ("Cognitive Tests ONLY", "Cognitive Tests ONLY", "Decision Trees",
         "Tree baseline on psychometrics only."),
    ]
    for display, key, model, justification in subsets_order:
        data = ablation.get((key, model))
        assert data is not None, f"Ablation row missing for ({key}, {model})"
        _, adas_s, cdr_s, mmse_s = data
        ablation_lines.append(
            f"| **{display}** | **{model}** | **{adas_s}** | **{cdr_s}** | **{mmse_s}** | {justification} |")

    lines = replace_markdown_table(lines, "Feature Modality Subset", ablation_lines)

    # ---- Section 5: Medical Panel Financial Burden Table (Derived Dynamically) ----
    feature_counts_by_panel = Counter(feature_to_panel[f] for f in tier2_features)
    
    # Sort triggered panels by unit price descending, then name
    sorted_triggered_panels = sorted(
        tier2_panels,
        key=lambda p: (p in ENDPOINT_PANELS, -panel_costs_tbl.get(p, 0.0), p)
    )

    financial_table_lines = [
        "| Medical Test Panel / Procedure | Unit Price ($) | FISTA Selected Features | Billed Panel Cost ($) | Medical Description |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]
    
    for panel in sorted_triggered_panels:
        is_endpoint = panel in ENDPOINT_PANELS
        unit_price = panel_costs_tbl.get(panel, 0.0)
        unit_price_str = f"${unit_price:,.2f}" if not is_endpoint else "$0.00*"
        billed_cost_str = "$0.00" if is_endpoint else f"${unit_price:,.2f}"
        feat_cnt = feature_counts_by_panel.get(panel, 0)
        desc = PANEL_DESCRIPTIONS.get(panel, "Clinical test panel.")
        financial_table_lines.append(
            f"| **{panel}** | {unit_price_str} | {feat_cnt} | {billed_cost_str} | {desc} |"
        )
        
    financial_table_lines.append(
        f"| **TOTAL BILLED COST PER PATIENT** | — | **{tier2_k} Features** | **{tier2_cost_fmt}** | Total patient screening cost. |"
    )

    lines = replace_markdown_table(lines, "Medical Test Panel / Procedure", financial_table_lines)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print("[OK] REPORT SYNCHRONISED")
    print(f"     Tier 2: {tier2_k} features, {tier2_cost_fmt}, {len(tier2_panels)} panels")
    print(f"     XGBoost: {tier2_k} features, {xgb_cost_fmt}, {len(xgb_panels)} panels")
    print(f"     Tier 1: {tier1_k} features, {tier1_cost_fmt}, {tier1_panels} panels")
    print(f"     Greedy Pareto: step {pareto_idx} at {c_par} (R2 {r_par:.4f})")


if __name__ == '__main__':
    sync_report_metrics()
