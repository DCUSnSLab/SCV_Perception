# object_depth_tracker/filters/__init__.py
def build(filter_name: str):
    """
    filter_name: 'centroid' | 'kalman6d' | 'sort3d'
    returns instance with .update(meas_list, stamp) -> list[Track]
    """
    if filter_name in ("centroid", "kalman_6d", "sort3d"):
        module = __import__(f"filters.{filter_name}",
                            fromlist=[""])
        return module.Filter()
    raise ValueError(f"Unknown filter {filter_name}")