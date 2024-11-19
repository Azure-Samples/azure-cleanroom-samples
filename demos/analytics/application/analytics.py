from flask import Flask, jsonify, request
from pyspark.sql import SparkSession
from pyspark import SparkFiles
import os
import logging
import requests

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)
app = Flask(__name__)

# Initialize global variables
spark = None
combined_df = None
storage_paths = []


# Establish Spark session and load CSV files
def initialize_spark_session():
    global spark, combined_df

    try:
        spark = SparkSession.builder.appName("COUNT_MENTIONS").getOrCreate()

        # Retrieve storage paths from environment variables
        i = 1
        while True:
            env_var = f"STORAGE_PATH_{i}"
            storage_path_env = os.getenv(env_var)
            if storage_path_env:
                storage_paths.append(storage_path_env)
                i += 1
            else:
                break

        # Combine CSV files from multiple storage paths
        for storage_path in storage_paths:
            if os.path.exists(storage_path):
                files = [
                    file for file in os.listdir(storage_path) if file.endswith(".csv")
                ]
                if files:
                    csv_files = [os.path.join(storage_path, file) for file in files]
                    for file_path in csv_files:
                        spark.sparkContext.addFile(file_path)
                        df = spark.read.csv(
                            SparkFiles.get(os.path.basename(file_path)),
                            header=True,
                            inferSchema=True,
                        )
                        if combined_df is None:
                            combined_df = df
                        else:
                            combined_df = combined_df.union(df)

        if combined_df is not None:
            headers = ["date", "time", "author", "mentions"]
            combined_df = combined_df.toDF(*headers)
            combined_df.createOrReplaceTempView("COMBINED_TWEETS")
            logging.info("Spark session and data initialization successful")
        else:
            logging.warning("No CSV files found or loaded into combined_df")

    except Exception as e:
        logging.error(f"Error initializing Spark session: {e}")
        spark = None
        combined_df = None


def get_document_from_cgs(queryId):
    governanceEndpoint = "http://localhost:8300"
    uri = f"{governanceEndpoint}/documents/{queryId}"

    logging.info(f"Sending query request to uri: {uri}")
    resp = requests.post(
        url=uri,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


# Route handler for running Spark queries
@app.route("/app/run_query/<documentID>", methods=["GET"])
def run_spark(documentID):
    try:
        global spark, combined_df

        if spark is None or combined_df is None:
            return (
                jsonify({"error": "Spark session or data not initialized properly."}),
                500,
            )

        # Get document from CGS
        document = get_document_from_cgs(documentID)
        logging.info(f"Document retrieved: {document}")

        if "state" in document and document["state"] != "Accepted":
            return (
                jsonify({"error": "Document state is not 'Accepted'."}),
                400,
            )

        query = document["data"]

        # Execute SQL query
        result = spark.sql(query)
        result_list = result.collect()
        logging.info("Query result generated")

        # Convert query result to JSON format and return as response
        result_dicts = [row.asDict() for row in result_list]
        return jsonify(result_dicts)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Initialize Spark session and load data
    initialize_spark_session()
    app.run(host="0.0.0.0", port=8310)
    os.system("sleep infinity")
