import os
import csv
import re
import argparse

ADMIN_REGEX = re.compile(r'^(SITEID(\..*)?|IMAGEUID(\..*)?|STATUS(\..*)?|VERSION(\..*)?|LONIUID(\..*)?|SOURCE(\..*)?|RAWQC|qc_flag|FSVER|FIELD_STRENGTH|MANUFACTURER|TRACER|TRACER_SUVR_WARNING|BATCH|KIT|STDS|USERDATE|update_stamp|HAS_QC_ERROR|DD_CRF_VERSION|VISCODE|VISDATE|ORIGPROT|COLPROT|EXAMDATE|RUNDATE|DRAWDTE|RID|ID)$', re.IGNORECASE)

def evaluate_panel_costs(panel_costs_path, feature_mapping_path, selected_features_path):
    print(f"Loading panel costs from: {panel_costs_path}")
    panel_costs = {}
    with open(panel_costs_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            panel_costs[row['Panel_Name']] = float(row['Cost_USD'])
            
    print(f"Loading feature mapping table from: {feature_mapping_path}")
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
                feat = row[feature_col_idx]
                if not ADMIN_REGEX.match(feat):
                    selected_features.append(feat)
                
    print(f"Loaded {len(selected_features)} non-administrative clinical features.")
    
    # Evaluate costs
    triggered_panels = set()
    unmapped_features = []
    
    for feature in selected_features:
        panel_name = feature_to_panel.get(feature, 'UNMAPPED')
        
        if panel_name == 'UNMAPPED':
            unmapped_features.append(feature)
        else:
            triggered_panels.add(panel_name)
            
    if unmapped_features:
        print(f"\nWARNING: {len(unmapped_features)} selected features are labeled as 'UNMAPPED'!")
        print(f"Examples: {unmapped_features[:10]}")
    
    # Calculate totals
    total_cost = 0
    panel_breakdown = []
    
    for panel_name in triggered_panels:
        cost = panel_costs.get(panel_name, 0.0)
        total_cost += cost
        panel_breakdown.append((panel_name, cost))
            
    print("\n" + "="*50)
    print("COST EVALUATION RESULTS")
    print("="*50)
    print(f"Total Unique Panels Triggered: {len(triggered_panels)}")
    print(f"Total Estimated Financial Cost: ${total_cost:,.2f}")
    print("\nTriggered Panels Breakdown:")
    for name, cost in panel_breakdown:
        feat_count = sum(1 for f in selected_features if feature_to_panel.get(f) == name)
        print(f"  - {name}: ${cost:,.2f} ({feat_count} clinical features used)")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clinical trial panel costs.")
    parser.add_argument("--panel_costs", type=str, required=True, help="Path to panel_costs.csv (Panel_Name, Cost_USD)")
    parser.add_argument("--mapping_csv", type=str, required=True, help="Path to feature_to_panel_mapping.csv (Feature_Name, Panel_Name)")
    parser.add_argument("--features_csv", type=str, required=True, help="Path to selected_features.csv")
    
    args = parser.parse_args()
    evaluate_panel_costs(args.panel_costs, args.mapping_csv, args.features_csv)
