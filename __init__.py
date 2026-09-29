import os

import folder_paths
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .nodes import LoadTypedDecisionModel, TypedDecision

folder_paths.add_model_folder_path("typed_decision", os.path.join(folder_paths.models_dir, "typed_decision"))


class TypedDecisionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadTypedDecisionModel, TypedDecision]


async def comfy_entrypoint() -> TypedDecisionExtension:
    return TypedDecisionExtension()
