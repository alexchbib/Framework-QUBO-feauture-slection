import csv

# Mandatory trial outcome instruments — billed at $0 per trial budget policy.
ENDPOINT_PANELS = {
    "ADAS-Cog Assessment",
    "Clinical Dementia Rating (CDR)",
    "MMSE Assessment",
}


def load_panel_tables(costs_path, mapping_path):
    """Loads panel_costs.csv and feature_to_panel_mapping.csv."""
    panel_costs = {}
    with open(costs_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            panel_costs[row['Panel_Name']] = float(row['Cost_USD'])

    feature_to_panel = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            feature_to_panel[row['Feature_Name']] = row['Panel_Name']

    return panel_costs, feature_to_panel


def billed_cost_for_features(feature_names, panel_costs, feature_to_panel,
                             include_endpoint_costs=False):
    """Panel-level billed cost for a selected feature set.
    Mirrors evaluate_costs.py exactly. Returns (total_cost, triggered_panels)."""
    triggered = set()
    for feat in feature_names:
        assert feat in feature_to_panel, \
            f"Feature '{feat}' is missing from feature_to_panel mapping table!"
        triggered.add(feature_to_panel[feat])

    total = 0.0
    for panel in triggered:
        if not include_endpoint_costs and panel in ENDPOINT_PANELS:
            continue
        total += panel_costs.get(panel, 0.0)
    return total, triggered


def read_selected_features(csv_path):
    """Reads the Selected_Feature column out of a selected_features_*.csv."""
    names = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = header.index('Selected_Feature') if 'Selected_Feature' in header else 0
        for row in reader:
            if row:
                names.append(row[idx])
    return names
