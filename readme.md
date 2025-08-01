# Intelligent Traditional Chinese Medicine Prescription Recommendation System Combined with Survival Analysis

- ## datasets

  Contains all data files.

- ## mode

  Contains the trained MLKNN and KNN models.

- ## static

  > ### **cluster**
  >
  > > Population division result data.
  >
  > ### **CSS**
  >
  > > All CSS code.
  >
  > ### **images**
  >
  > > Image resources.
  >
  > ### **js**
  >
  > > JavaScript files.
  >
  > ### **uploads**
  >
  > > User-uploaded dataset files.

- ## templates

  **All front-end code.**

  - **base.html**

    Base template, navigation bar functionality.

  - **dataset.html**

    Dataset management front-end.

  - **home.html**

    Platform homepage.

  - **index.html**

    Welcome interface.

  - **input.html**

    Consultation information collection.

  - **log.html**

    Diagnosis log.

  - **login.html**

    Login interface.

  - **model.html**

    Model management.

  - **other_home.html**

    Others' homepage.

  - **output.html**

    Recommendation results.

  - **people_divide.html**

    Population division.

  - **people_divide_info.html**

    Detailed population information.

  - **report.html**

    Diagnosis report.

  - **signup.html**

    Signup interface.

  - **user_home.html**

    User homepage.

- ## Python files

  - **algorithm.py**

    Contains MLKNN and KNN algorithms.

  - **Clustering2Analysis.py**

    Handles population division.

  - **data_check.py**

    Performs data format checks for population division.

  - **jaccard.py**

    Calculates similar populations.

  - **main.py**

    The Flask backend.

  - **PR_system_sim.py**

    The similarity recommendation algorithm.

  - **similarity.py**

    Finds similar classic prescriptions.
