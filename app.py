import streamlit as st
import numpy as np
import pandas as pd
import scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the dataset (assuming you have it saved in a CSV file)
@st.cache_data  # Use st.cache for data caching
def load_data():
    df = pd.read_csv("/Users/shreyamahajan/Desktop/college/dslproj/preprocessed_data.csv", encoding='utf-8')
    return df

df_train = load_data()

# Define the main Streamlit app
def main():
    st.title("Genetic Disorder Prediction App")

    # Sidebar for user input
    st.sidebar.header("User Input")

    selected_features = st.sidebar.multiselect("Select Features", df_train.columns, default=["Status"])
    st.sidebar.subheader("Please select Disorder Subclass for prediction.")

    # Check if both 'Status' and 'Disorder Subclass' are in selected features
    if 'Status' in selected_features:
        selected_features = [col for col in selected_features if col not in ['Status']]
        

        # Ensure at least one feature (excluding 'Status' and 'Disorder Subclass') is selected
        if not selected_features:
            st.error("Please select at least one feature (excluding 'Status').")
        else:
            # Filter the dataset based on selected features
            filtered_df = df_train[selected_features]

            # Split the dataset into features (X) and target (y)
            X = df_train[selected_features]
            y = df_train['Status']

            # Filter out non-numeric features
            numeric_features = [feature for feature in selected_features if df_train[feature].dtype.kind in ('i', 'f')]
            X = X[numeric_features]

            # Convert X to int
            X = X.astype(int)

            # Handle missing values in the target variable 'y_train'
            y = y.fillna(y.mode().values[0])

            # Convert y_train to a NumPy array
            y_train = y.values

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.3, random_state=1)

            # Model training (you can choose your own model)
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)

            # User input for prediction
            st.subheader("Predict Genetic Disorder status in the offspring:")
            st.write("Enter 1 for Yes and 0 for No")
            user_input = {}

            for feature in numeric_features:
                min_value = 0
                max_value = 1

                if feature in ['Birth asphyxia', 'Autopsy shows birth defect (if applicable)', 'H/O serious maternal illness', 'H/O radiation exposure (x-ray)', 'H/O substance abuse']:
                    user_input[feature] = st.selectbox(f"Enter {feature}", ['0', '1'])
                elif feature in ['Disorder Subclass']:
                    user_input[feature] = st.selectbox(f"Enter {feature}",  ['0','1','2','3','4','5','6','7','8'])
                elif feature in ["Patient Age", "Mother's Age", "Father's Age"]:
                    user_input[feature] = st.number_input(f"Enter {feature}", min_value=0, max_value=100)
                elif feature in ['Blood cell count (mcL)']:
                    user_input[feature] = st.number_input(f"Enter {feature}", min_value=0.0, max_value=10.0, format="%.7f")

                elif feature in ['Maternal gene', 'Paternal gene']:
                    user_input[feature] = st.selectbox(f"Enter {feature}", ['0', '1'])
                    st.write("(Enter 1 for Present or 0 for Absent)")
                elif feature in ['Gender']:
                    user_input[feature] = st.selectbox(f"Enter {feature}", ['1', '0'])
                    st.write("(Enter 1 for Male or 0 for Female)")
                else:
                    user_input[feature] = st.number_input(f"Enter {feature}", min_value=0, max_value=100)

                if feature in ['Disorder Subclass']:
                    st.write("0: Cystic fibrosis")
                    st.write("1: Leber's hereditary optic neuropathy")
                    st.write("2: Diabetes")
                    st.write("3: Leigh syndrome")
                    st.write("4: Cancer")
                    st.write("5: Tay-Sachs")
                    st.write("6: Hemochromatosis")
                    st.write("7: Mitochondrial myopathy")
                    st.write("8: Alzheimer's")
                    
                    info_toggle = st.sidebar.checkbox("Click here for information about all diseases")

                    if info_toggle:
                        st.sidebar.markdown("**Cystic fibrosis:** Cystic fibrosis is a genetic disorder that affects the lungs and digestive system. It leads to the production of thick and sticky mucus, causing breathing difficulties and recurrent lung infections.")
                        st.sidebar.markdown("**1: Leber's hereditary optic neuropathy:** Leber's hereditary optic neuropathy is a rare mitochondrial disorder that primarily affects the optic nerve, leading to vision loss.")
                        st.sidebar.markdown("**2: Diabetes:** Diabetes is a chronic metabolic condition that affects blood sugar levels. It can result in high blood sugar (hyperglycemia) or low blood sugar (hypoglycemia) and can lead to various complications if not managed properly.")
                        st.sidebar.markdown("**3: Leigh syndrome:** Leigh syndrome is a severe neurological disorder that typically appears in infancy or early childhood. It is characterized by progressive loss of motor skills, muscle weakness, and other neurological symptoms.")
                        st.sidebar.markdown("**4: Cancer:** Cancer is a group of diseases characterized by uncontrolled cell growth and the potential to invade or spread to other parts of the body. There are various types of cancer, each with its own characteristics and treatments.")
                        st.sidebar.markdown("**5: Tay-Sachs:** Tay-Sachs disease is a rare genetic disorder that affects the nervous system. It results in the accumulation of harmful substances in the brain, leading to severe neurological problems and a shortened lifespan.")
                        st.sidebar.markdown("**6: Hemochromatosis:** Hemochromatosis is a hereditary condition that causes the body to absorb and store too much iron from the diet. Excess iron can accumulate in various organs and lead to complications.")
                        st.sidebar.markdown("**7: Mitochondrial myopathy:** Mitochondrial myopathy refers to a group of neuromuscular disorders caused by problems with the mitochondria, the energy-producing structures in cells. Symptoms may include muscle weakness and fatigue.")
                        st.sidebar.markdown("**8: Alzheimer's:** Alzheimer's disease is a progressive brain disorder that affects memory, thinking, and behavior. It is the most common cause of dementia in older adults and leads to cognitive decline and memory loss.")



                    if st.button("Predict"):
                        if 'Disorder Subclass' in user_input:
                            input_data = pd.DataFrame([user_input])
                            prediction = knn.predict(input_data)
                            status = 'Present' if prediction[0] == 1 else 'Absent'
                            st.write(f"The gene for the disorder to occur is: {status}")
                        else:
                            st.error("Please select 'Disorder Subclass',' 'Maternal Gene', and 'Paternal Gene'  for prediction.")


if __name__ == "__main__":
    main()
