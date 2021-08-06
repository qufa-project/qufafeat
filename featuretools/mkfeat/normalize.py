from pandas import DataFrame
import autonormalize as an


def normalize(df: DataFrame, key_colname):
    if len(df) > 1000:
        df = df.sample(n=1000)
    es = an.auto_entityset(df, index=key_colname, accuracy=0.98)

    norminfos = []
    # 첫번째 이외의 entity들에 대해서. 첫번째 entity가 main임을 가정
    entities = es.entities[1:]
    for et in entities:
        norminfo = []
        for var in et.variables:
            norminfo.append(var.name)
        norminfos.append(norminfo)
    for norminfo in norminfos:
        parent_ids = _get_parent_entity_ids(es, norminfo[0])
        for parent_id in parent_ids:
            vars = es[parent_id].variables
            for var in vars[1:]:
                norminfo.append(var.name)
    return norminfos


def _get_parent_entity_ids(es, child_id):
    parent_ids = []
    for rel in es.relationships:
        if child_id == rel.child_entity.id:
            parent_ids.append(rel.parent_entity.id)
            parent_ids += _get_parent_entity_ids(es, rel.parent_entity.id)
    return parent_ids
