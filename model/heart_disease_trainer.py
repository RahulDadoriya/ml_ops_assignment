import os
import matplotlib
# Force a headless background backend so graphics operations do not block the pipeline execution
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class HeartDiseaseTrainer:
    """
    Core MLOps & Training engine.
    Handles data splitting, EDA tracking, and model serialization via MLf
    """

    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initializes the tracking environment context securely.
        Forces the MLflow engine to bind to the SQLite layer and sets local artifact files.
        """
        import os

        # 1. Clear out variable overrides and enforce tracking destinations
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # 2. Establish tracking framework target configuration locations
        mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        # 4. Enforce structural pickling globally to stop skops verification errors
        mlflow.sklearn.autolog(serialization_format="pickle", log_models=True)

    def process_and_split_data(self, df_clean, target_column="target"):
        """
        Splits features and targets safely using flat 1D target arrays.
        """
        X = df_clean.drop(columns=[target_column])

        # FIX: Force the target series to flatten to a clean 1D array right before splitting
        y = df_clean[target_column].values.ravel()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def generate_and_log_eda(self, df):
        """
        Executes Exploratory Data Analysis silently and uploads images as distinct MLflow run artifacts.
        """
        with mlflow.start_run(run_name="Dataset_EDA"):
            print("Logging EDA artifacts directly to local tracking storage...")

            # 1. Class Distribution
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="target", ax=ax1)
            plt.xticks(rotation=20)
            plt.tight_layout()
            mlflow.log_figure(fig1, "eda/class_distribution.png")
            plt.close(fig1)

            # 2. Feature Distributions
            df.drop(columns=["target"]).hist(figsize=(10, 7))
            plt.tight_layout()
            fig2 = plt.gcf()
            mlflow.log_figure(fig2, "eda/feature_distributions.png")
            plt.close(fig2)

            # 3. Relationships/Pairplot
            pairplot_obj = sns.pairplot(df, hue="target")
            plt.tight_layout()
            mlflow.log_figure(pairplot_obj.figure, "eda/feature_pairplot.png")
            plt.close(pairplot_obj.figure)

            # 4. Correlation Analysis Matrix
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.drop(columns=["target"]).corr(), annot=True, cmap="coolwarm", ax=ax4)
            plt.tight_layout()
            mlflow.log_figure(fig4, "eda/feature_correlation.png")
            plt.close(fig4)

            print("EDA figures successfully archived.")

    def run_training_pipeline(self, X_train, X_test, y_train, y_test):
        """
        Iterates over the defined model architectures using robust Pipeline configurations.
        Returns a dictionary mapping the Model Names to their unique, production-ready Run IDs.
        """
        # Construct production-grade architecture blueprints using Pipeline encapsulation
        models = [
            (
                "Logistic Regression",
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
                ])
            ),
            (
                "Random Forest",
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
            ),
        ]

        run_registry = {}

        for model_name, pipeline in models:
            # Open a cleanly tracked parent structural run context locally
            with mlflow.start_run(run_name=model_name) as run:
                print(f"Executing training loop for architecture: {model_name}")

                # Fit the operational model architecture pipeline seamlessly
                pipeline.fit(X_train, y_train)

                # Evaluate model validation metrics
                y_pred = pipeline.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)

                # Sync specific operational parameter and tracking targets
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("accuracy", report["accuracy"])
                mlflow.log_metric("recall_class_1", report["1"]["recall"])
                mlflow.log_metric("recall_class_0", report["0"]["recall"])
                mlflow.log_metric("f1_score_macro", report["macro avg"]["f1-score"])

                # Register the successful run mapping id to memory
                run_registry[model_name] = run.info.run_id
                print(f"Logged {model_name} with Tracking Run ID: {run.info.run_id}")

        return run_registry