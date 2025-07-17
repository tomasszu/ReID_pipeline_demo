import lancedb
import pyarrow as pa
import numpy as np
import shutil

class LanceDBOperator:
    def __init__(self, db_path, table_name="vehicle_features", features_size=256):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name

        schema = pa.schema([
            pa.field("vehicle_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), features_size)),
            pa.field("times_summed", pa.int8())
        ])

        if table_name not in self.db.table_names():
            self.table = self.db.create_table(table_name, data=None, schema=schema)
            print(f"[LanceDB] Created table '{table_name}' with feature size {features_size}")
        else:
            self.table = self.db.open_table(table_name)
            print(f"[LanceDB] Opened existing table '{table_name}'")

    def add_features(self, features_with_ids):
        """
        Args:
            features_with_ids: List of tuples (vehicle_id: str/int, feature_vector: np.ndarray or list)
        """
        
        if len(features_with_ids) == 0 or features_with_ids is None:
            return
        
        records = [
            {
                "vehicle_id": str(obj_id),
                "vector": feature.tolist() if hasattr(feature, "tolist") else feature,
                "times_summed": 1  # default, you can increment this later if needed
            }
            for obj_id, feature in features_with_ids
        ]
        self.table.add(records)

    def delete_table(self):
        """Permanently delete the table directory from disk."""
        table_path = os.path.join(self.db_path, self.folder)
        if os.path.exists(table_path):
            shutil.rmtree(table_path)
            print(f"Deleted table and data at: {table_path}")
        else:
            print(f"Table path does not exist: {table_path}")
