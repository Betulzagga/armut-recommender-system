
"""
Association Rule Learning Based Recommender System for Armut.com

This script builds a service recommendation system using association rules
from user transaction data provided by Armut, a major service marketplace in Turkey.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Configure display options
pd.set_option('display.max_columns', None)

# -------------------------------
# TASK 1: Data Preparation
# -------------------------------

# Step 1: Read the dataset
df = pd.read_csv("armut_data.csv")

# Step 2: Create a unique service identifier by combining ServiceId and CategoryId
df["Service"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

# Step 3: Convert CreateDate to datetime and generate monthly basket identifiers
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["Month"] = df["CreateDate"].dt.strftime("%Y-%m")
df["BasketID"] = df["UserId"].astype(str) + "_" + df["Month"]

# -------------------------------
# TASK 2: Generate Association Rules
# -------------------------------

# Step 1: Create the basket-service pivot table
basket_service_df = df.groupby(["BasketID", "Service"])["Service"]                       .count().unstack().fillna(0)

# Convert service counts to binary indicators (0 or 1)
basket_service_df = basket_service_df.map(lambda x: 1 if x > 0 else 0)

# Step 2: Generate frequent itemsets using Apriori
frequent_itemsets = apriori(basket_service_df, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

# -------------------------------
# Recommendation Functions
# -------------------------------

def arl_recommender(rules_df: pd.DataFrame, product_id: str, rec_count: int = 1) -> list:
    """
    Recommend related services based on the association rules.

    Parameters:
    -----------
    rules_df : pd.DataFrame
        DataFrame containing association rules.

    product_id : str
        The service ID for which recommendations are to be generated.

    rec_count : int, optional
        Number of recommendations to return (default is 1).

    Returns:
    --------
    List[str]
        List of recommended service IDs.
    """
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, products in sorted_rules["antecedents"].items():
        for item in products:
            if item == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    return get_recommendations(recommendation_list, rec_count)

def get_recommendations(recommendation_list: list, rec_count: int) -> list:
    """
    Clean and flatten the recommendation list and return top N unique items.

    Parameters:
    -----------
    recommendation_list : list
        Nested list of recommended items.

    rec_count : int
        Number of top recommendations to return.

    Returns:
    --------
    List[str]
        List of unique recommended service IDs.
    """
    flat_unique_recommendations = list({item for sublist in recommendation_list for item in sublist})
    return flat_unique_recommendations[:rec_count]


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Recommend services for a user who received service "2_0"
    recommended_services = arl_recommender(rules, "2_0", rec_count=4)
    print("Recommended Services for 2_0:", recommended_services)
