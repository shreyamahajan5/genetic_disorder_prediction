
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_data.csv", encoding='utf-8')
    return df

df_train = load_data()

def main():
    st.title("Genetic Disorder Prediction App")

    # Sidebar for user input
    st.sidebar.header("User Input")

    # Always include 'Disorder Subclass' in the features
    default_features = ["Disorder Subclass"]
    available_features = [col for col in df_train.columns if col not in ["Status", "Disorder Subclass"]]
    selected_features = st.sidebar.multiselect("Select Additional Features", available_features)

    # Final selected features
    selected_features = default_features + selected_features

    # Ensure at least one other feature is selected
    if len(selected_features) <= 1:
        st.error("Please select at least one additional feature along with 'Disorder Subclass'.")
    else:
        filtered_df = df_train[selected_features]
        X = df_train[selected_features]
        y = df_train["Status"].fillna(df_train["Status"].mode().values[0])  # fill missing in target

        numeric_features = [feature for feature in selected_features if df_train[feature].dtype.kind in ('i', 'f')]
        X = X[numeric_features].astype(int)
        y_train = y.values

        X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.3, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        st.subheader("Predict Genetic Disorder Status in the Offspring")
        st.write("Enter 1 for Yes and 0 for No")

        user_input = {}
        for feature in numeric_features:
            if feature == "Disorder Subclass":
                user_input[feature] = st.selectbox(f"Enter {feature}", list(range(9)))
                st.write("0: Cystic fibrosis")
                st.write("1: Leber's hereditary optic neuropathy")
                st.write("2: Diabetes")
                st.write("3: Leigh syndrome")
                st.write("4: Cancer")
                st.write("5: Tay-Sachs")
                st.write("6: Hemochromatosis")
                st.write("7: Mitochondrial myopathy")
                st.write("8: Alzheimer's")
            elif feature in ['Birth asphyxia', 'Autopsy shows birth defect (if applicable)', 'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse']:
                user_input[feature] = st.selectbox(f"Enter {feature}", ['0', '1'])
            elif feature in ["Patient Age", "Mother's Age", "Father's Age"]:
                user_input[feature] = st.number_input(f"Enter {feature}", min_value=0, max_value=100)
            elif feature in ['Blood cell count (mcL)']:
                user_input[feature] = st.number_input(f"Enter {feature}", min_value=0.0, max_value=10.0, format="%.7f")
            elif feature in ['Maternal gene', 'Paternal gene']:
                user_input[feature] = st.selectbox(f"Enter {feature}", ['0', '1'])
                st.write("(Enter 1 for Present or 0 for Absent)")
            elif feature == 'Gender':
                user_input[feature] = st.selectbox("Enter Gender", ['1', '0'])
                st.write("(Enter 1 for Male or 0 for Female)")
            else:
                user_input[feature] = st.number_input(f"Enter {feature}", min_value=0, max_value=100)

        if st.sidebar.checkbox("Click here for information about all diseases"):
            with st.sidebar:
                st.markdown("**0: Cystic fibrosis** – Affects lungs and digestion.")
                st.markdown("**1: Leber's hereditary optic neuropathy** – Causes vision loss.")
                st.markdown("**2: Diabetes** – Metabolic disorder affecting blood sugar.")
                st.markdown("**3: Leigh syndrome** – Severe neurological disorder.")
                st.markdown("**4: Cancer** – Uncontrolled cell growth.")
                st.markdown("**5: Tay-Sachs** – Rare disorder affecting the brain.")
                st.markdown("**6: Hemochromatosis** – Excess iron buildup.")
                st.markdown("**7: Mitochondrial myopathy** – Muscle/energy production issues.")
                st.markdown("**8: Alzheimer's** – Memory and cognitive decline.")

        if st.button("Predict"):
            input_data = pd.DataFrame([user_input])
            prediction = knn.predict(input_data)
            status = "Present" if prediction[0] == 1 else "Absent"
            st.success(f"The gene for the disorder to occur is: **{status}**")

if __name__ == "__main__":
    main()
