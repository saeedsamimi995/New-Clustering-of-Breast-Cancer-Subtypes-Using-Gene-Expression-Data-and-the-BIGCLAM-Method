"""Utilities for comparing BIGCLAM communities with PAM50 labels."""

import pandas as pd


def compare_with_pam50(clustered_data_path, pam50_data_path):
    """
    Compare BIGCLAM community assignments with PAM50 labels.
    
    Args:
        clustered_data_path (str): Path to CSV with BIGCLAM community assignments.
        pam50_data_path (str): Path to CSV with PAM50 labels.
        
    Returns:
        pd.DataFrame: Comparison results showing label counts per community.
    """
    # Load the clustered dataset
    data_with_community = pd.read_csv(clustered_data_path)
    
    # Load dataset containing PAM50 column as target
    data = pd.read_csv(pam50_data_path)
    data = data.T
    
    # Map PAM50 labels to numeric values
    pam50_mapping = {
        'pam50 subtype: Normal': 0,
        'pam50 subtype: LumA': 1,
        'pam50 subtype: LumB': 2,
        'pam50 subtype: Her2': 3,
        'pam50 subtype: Basal': 4
    }
    
    for old_label, new_label in pam50_mapping.items():
        data.iloc[:, -1] = data.iloc[:, -1].replace(old_label, new_label)
    
    # Remove Normal samples
    data = data[data.iloc[:, -1] != 0]
    label_data = data.iloc[1:, -1]
    
    # Assign the labels to the last column of data_with_community
    data_with_community['label'] = label_data.values
    
    # Ensure the columns are named correctly
    if 'Community' not in data_with_community.columns:
        data_with_community.rename(
            columns={data_with_community.columns[-2]: 'Community'}, 
            inplace=True
        )
    if 'label' not in data_with_community.columns:
        data_with_community.rename(
            columns={data_with_community.columns[-1]: 'label'}, 
            inplace=True
        )
    
    # Count the number of each label in each community
    label_counts_per_community = (
        data_with_community.groupby('Community')['label']
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
    )
    
    # Display the results
    print("Comparison of BIGCLAM Communities with PAM50 Labels:")
    print("=" * 60)
    for community, labels in label_counts_per_community.iterrows():
        print(f"\nCommunity {community}:")
        for label, count in labels.items():
            print(f"  PAM50 class {int(label)}: {count} samples")
    
    return label_counts_per_community

