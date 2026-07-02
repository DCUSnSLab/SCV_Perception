# object_depth_tracker/filters/__init__.py
def build(filter_name: str, **kwargs):
    """
    filter_name: 'centroid' | 'kalman_6d' | 'kalman_6d_v2' | 'sort3d'
    returns instance with .update(meas_list, stamp) -> list[Track]
    """
    if filter_name in ("centroid", "kalman_6d", "kalman_6d_v2", "sort3d"):
        try:
            # Try relative import first
            module = __import__(f"object_depth_tracker.filters.{filter_name}",
                                fromlist=[""])
        except ImportError:
            # Fall back to direct import
            module = __import__(f"filters.{filter_name}",
                                fromlist=[""])
        
        # Pass parameters to filters that support them
        if filter_name in ("kalman_6d", "kalman_6d_v2") and kwargs:
            return module.Filter(**kwargs)
        else:
            return module.Filter()
    raise ValueError(f"Unknown filter {filter_name}")