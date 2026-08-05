import os
import sys
import csv
import argparse

def evaluate_panel_costs(panel_costs_path, feature_mapping_path, selected_features_path, include_endpoint_costs=False):
    print(f"Loading panel costs from: {panel_costs_path}")
    panel_costs = {}
    with open(panel_costs_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            panel_costs[row['Panel_Name']] = float(row['Cost_USD'])
            
    print(f"Loading feature mapping table (provenance) from: {feature_mapping_path}")
    feature_to_panel = {}
    with open(feature_mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature_to_panel[row['Feature_Name']] = row['Panel_Name']
    
    print(f"Loading selected features from: {selected_features_path}")
    selected_features = []
    with open(selected_features_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_col_idx = 0
        if 'Selected_Feature' in header:
            feature_col_idx = header.index('Selected_Feature')
            
        for row in reader:
            if row:
                selected_features.append(row[feature_col_idx])
                
    print(f"Loaded {len(selected_features)} selected clinical features.")
    
    # Endpoint mandatory instruments (Fixes B7)
    endpoint_panels = {"ADAS-Cog Assessment", "Clinical Dementia Rating (CDR)", "MMSE Assessment"}
    
    triggered_panels = set()
    unmapped_features = []
    
    for feature in selected_features:
        panel_name = feature_to_panel.get(feature, 'UNMAPPED')
        if panel_name == 'UNMAPPED':
            unmapped_features.append(feature)
        else:
            triggered_panels.add(panel_name)
            
    if unmapped_features:
        print(f"\nWARNING: {len(unmapped_features)} selected features are unmapped!")
        print(f"Examples: {unmapped_features[:5]}")
    
    total_cost = 0.0
    panel_breakdown = []
    
    for panel_name in sorted(triggered_panels):
        cost = panel_costs.get(panel_name, 0.0)
        
        # Mandatory trial endpoint policy (Fixes B7)
        if not include_endpoint_costs and panel_name in endpoint_panels:
            billed_cost = 0.0
            note = " (Mandatory Trial Endpoint - Billed $0.00)"
        else:
            billed_cost = cost
            note = ""
            
        total_cost += billed_cost
        feat_count = sum(1 for f in selected_features if feature_to_panel.get(f) == panel_name)
        panel_breakdown.append((panel_name, cost, billed_cost, feat_count, note))
            
    print("\n" + "="*65)
    print("AUDITED COST EVALUATION RESULTS")
    print("="*65)
    print(f"Total Unique Panels Triggered: {len(triggered_panels)}")
    print(f"Total Billed Financial Burden per Patient: ${total_cost:,.2f}")
    if not include_endpoint_costs:
        print("Note: Endpoint cognitive batteries (ADAS, CDR, MMSE) billed at $0 (mandatory outcome measures).")
    print("\nTriggered Panels Breakdown:")
    for name, raw_cost, billed_cost, feat_count, note in panel_breakdown:
        print(f"  - {name}: ${billed_cost:,.2f}{note} ({feat_count} clinical features used)")
    print("="*65)
    return total_cost, triggered_panels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clinical trial panel costs.")
    parser.add_argument("--panel_costs", type=str, required=True, help="Path to panel_costs.csv")
    parser.add_argument("--mapping_csv", type=str, required=True, help="Path to feature_to_panel_mapping.csv")
    parser.add_argument("--features_csv", type=str, required=True, help="Path to selected_features.csv")
    parser.add_argument("--include_endpoint_costs", action="store_true", help="Charge endpoint cognitive tests")
    
    args = parser.parse_args()
    evaluate_panel_costs(args.panel_costs, args.mapping_csv, args.features_csv, args.include_endpoint_costs)
