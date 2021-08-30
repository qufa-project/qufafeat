from pandas import DataFrame
import autonormalize as an

n_samples = 1000
accuracy = 0.98
N_ITERS = 4


def normalize(df: DataFrame, key_colname):
    if len(df) > n_samples * N_ITERS:
        norminfos = None
        for _ in range(N_ITERS):
            df_samp = df.sample(n=n_samples)
            norminfos_new = _get_norminfos(df_samp, key_colname)
            if norminfos_new:
                if norminfos is None:
                    norminfos = norminfos_new
                else:
                    _merge_norminfos(norminfos, norminfos_new)
        return norminfos
    else:
        return _get_norminfos(df, key_colname)


def _merge_norminfos(norminfos, norminfos_new):
    for norminfo_new in norminfos_new:
        if _has_norminfos(norminfos, norminfo_new[0]):
            norminfo_merged = [norminfos[0][0]]
            for merged in set(norminfos[0][1:]) & set(norminfo_new[1:]):
                norminfo_merged.append(merged)
            norminfos.pop(0)
            norminfos.insert(0, norminfo_merged)


def _has_norminfos(norminfos, key):
    for norminfo in norminfos:
        if norminfo[0] == key:
            _clear_norminfos_upto_key(norminfos, key)
            return True
    return False


def _clear_norminfos_upto_key(norminfos, key):
    idx = 0
    for norminfo in norminfos:
        if norminfo[0] == key:
            break
        idx += 1
    for _ in range(idx):
        norminfos.pop(0)


def _get_norminfos(df: DataFrame, key_colname):
    try:
        es = an.auto_entityset(df, index=key_colname, accuracy=accuracy)
    except KeyError:
        # Maybe autonormalize bug. It seems to have a problem in case of multi key normalization.
        return None

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
