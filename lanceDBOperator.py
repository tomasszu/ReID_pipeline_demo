import lancedb
import pyarrow as pa
import numpy as np
import shutil
import os

class LanceDBOperator:
    def __init__(self, db_path, table_name="vehicle_features", features_size=256):
        """ Initializes the LanceDBOperator with the specified database path and table name.
        Args:
            db_path (str): Path to the LanceDB database directory.
            table_name (str): Name of the table to store vehicle features.
            features_size (int): Size of the feature vectors to be stored.
        """
        self.db = lancedb.connect(db_path)
        self.db_path = db_path
        self.table_name = table_name

        schema = pa.schema([
            pa.field("vehicle_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), features_size)),
            pa.field("times_summed", pa.int8()),
            pa.field("frame_added", pa.int32())  # NEW
        ])

        if table_name not in self.db.table_names():
            self.table = self.db.create_table(table_name, data=None, schema=schema)
            print(f"[LanceDB] Created table '{table_name}' with feature size {features_size}")
        else:
            self.table = self.db.open_table(table_name)
            print(f"[LanceDB] Opened existing table '{table_name}'")

    def add_features(self, features_with_ids, frame_id: int):
        """Adds a list of features to the LanceDB table.
        Args:
            features_with_ids (list): List of tuples where each tuple contains (vehicle_id, feature_vector).
            frame_id (int): The frame index when the features were extracted.
        """
        if len(features_with_ids) == 0 or features_with_ids is None:
            return
        
        records = [
            {
                "vehicle_id": str(obj_id),
                "vector": feature.tolist() if hasattr(feature, "tolist") else feature,
                "times_summed": 1,
                "frame_added": frame_id
            }
            for obj_id, feature in features_with_ids
        ]
        self.table.add(records)

    def query_features(self, features_with_ids):
        """
        Queries the LanceDB table for matching records based on the provided features.
        This method searches for the closest matches to the provided feature vectors.
        It returns a list of tuples containing the object ID, matched vehicle ID, and distance.
        Args:
            features_with_ids: List of tuples (vehicle_id: str/int, feature_vector: np.ndarray or list)
        
        Returns:
            List of matching records from the database
        """
        results = []

        if len(features_with_ids) == 0 or features_with_ids is None:
            return []
        
        for obj_id, feature in features_with_ids:

            df = self.table.search(feature) \
                .limit(1)  \
                .metric("Cosine") \
                .to_list()
            if df:
                match_id = df[0]["vehicle_id"]
                distance = df[0]["_distance"]

                # Only consider matches within a certain distance threshold
                # Adjust this threshold based on your requirements
                if distance <= 0.40:
                    results.append((obj_id, match_id, distance))
            else:
                print("No objects in database")
                return []

        
        return results

    def delete_table(self):
        """Permanently delete the table from disk."""

        # drop the table from the database
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
            print(f"[LanceDB] Deleted table '{self.table_name}'")

        
        # delete the database directory
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"[LanceDB] Deleted database directory '{self.db_path}'")

    def expire_old_features(self, current_frame, max_age):
        """Deletes features older than a certain number of frames.
        Args:
            current_frame (int): The current frame index.
            max_age (int): The maximum age of features to keep in the database.
        """
        try:
            self.table.delete(f"frame_added < {current_frame - max_age}")
            #print(f"[LanceDB] Deleted features older than {max_age} frames")
        except Exception as e:
            print(f"[LanceDB] Error during deletion: {e}")


