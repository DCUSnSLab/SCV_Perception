from .workspace_paths import default_bag_path
from .workspace_paths import default_image_topic
from .workspace_paths import default_model_path
from .workspace_paths import default_runtime_image_topic
from .workspace_paths import workspace_root


def main() -> None:
    root = workspace_root()
    ros2_bag = default_bag_path()
    model = default_model_path()

    print(f'workspace: {root}')
    print(f'bag: {ros2_bag}')
    print(f'bag image topic: {default_image_topic()}')
    print(f'runtime image topic: {default_runtime_image_topic()}')
    print(f'model: {model}')
