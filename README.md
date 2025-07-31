## 📦 Armut Recommender System

A market basket–based recommendation system built using Association Rule Learning for Armut, Turkey's leading online service marketplace. The system suggests services to users based on their historical monthly usage patterns.

---

### **🧠 Project Description**

This project leverages **Association Rule Learning (Apriori Algorithm)** to build a recommendation engine that suggests related services to users based on previously received services.

The dataset contains anonymized service usage data including:
- `UserId`: Unique customer identifier
- `ServiceId` and `CategoryId`: Representing specific services
- `CreateDate`: Timestamp of service usage

---

### **🚀 Features**
- Combines `ServiceId` and `CategoryId` to create a unique service token.
- Defines baskets based on `UserId` and monthly service usage.
- Applies the **Apriori algorithm** to discover frequent itemsets.
- Generates **association rules** with support, confidence, and lift.
- Includes a recommendation function for suggesting next services.

---

### **🛠️ Technologies Used**
- Python 3.x  
- pandas  
- mlxtend  
- Jupyter Notebook / .py Script

---

### **📂 How to Use**

```bash
# Clone the repository
git clone https://github.com/Betulzagga/armut-recommender-system.git
cd armut-recommender-system

# Open the script or Jupyter Notebook and run:
python armut_recommender.py
