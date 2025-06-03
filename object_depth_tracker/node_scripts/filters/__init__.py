# object_depth_tracker/filters/__init__.py
def build(filter_name: str):
    if filter_name in ("centroid", "kalman_6d", "kalman_6d_v2", "kalman_ca9d","sort3d"):
        module = __import__(f"filters.{filter_name}",
                            fromlist=[""])
        return module.Filter()
    raise ValueError(f"Unknown filter {filter_name}")