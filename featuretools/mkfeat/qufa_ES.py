from featuretools.entityset import EntitySet
import pandas as pd


class QufaES(EntitySet):
    def load_from_csv(self, path):
        data = pd.read_csv(path)
        self.entity_from_dataframe(entity_id="main", dataframe=data, index="id")


