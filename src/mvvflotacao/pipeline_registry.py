"""Project pipelines."""
from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.io import DataCatalog
from mvvflotacao.pipelines import data_processing, data_science, generate_models_inputs, models_optimization, optimization, check_raw_data

def register_pipelines() -> dict[str, Pipeline]:
    # """Register the project's pipelines.

    # Returns:
    #     A mapping from pipeline names to ``Pipeline`` objects.
    # """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    # return pipelines

    """Method that will be assigned to the callable returned by register_dynamic_pipelines(...), by a Hook."""
    raise NotImplementedError("""
        register_pipelines() is expected to be overwritten by ProjectHooks.
        Make sure the hooks is found in mvvflotacao/hooks and enabled in settings.py
        """)

def register_dynamic_pipelines(catalog: DataCatalog) -> dict[str, Pipeline]:
    """Register the project's pipelines depending on the catalog.

    Create pipelines dynamically based on parameters and datasets defined in the catalog.
    The function must return a callable without any arguments that will replace the
    `register_pipelines()` method in this same module, using an `after_catalog_created_hook`.

    Args:
        catalog: The DataCatalog loaded from the KedroContext.

    Returns:
        A callable that returns a mapping from pipeline names to ``Pipeline`` objects.
    """
    # create pipelines with access to catalog
    check_raw_data_pp = check_raw_data.create_pipeline(catalog = catalog)
    data_processing_pp = data_processing.create_pipeline(catalog = catalog)
    data_science_pp = data_science.create_pipeline(catalog = catalog)
    generate_models_inputs_pp = generate_models_inputs.create_pipeline(catalog = catalog)
    models_optimization_pp = models_optimization.create_pipeline(catalog = catalog)
    optimization_pp = optimization.create_pipeline(catalog = catalog)
    
    def register_pipelines():
        """Register the project's pipelines.

        Returns:
            A mapping from pipeline names to ``Pipeline`` objects.
        """
        pipelines = {
            "check_raw_data": check_raw_data_pp,
            "data_processing": data_processing_pp,
            "data_science": data_science_pp,
            "generate_models_inputs": generate_models_inputs_pp,
            "models_optimization": models_optimization_pp,
            "optimization": optimization_pp,
        }
        pipelines["__default__"] = data_processing_pp + generate_models_inputs_pp + data_science_pp
        pipelines["All"] = data_processing_pp + generate_models_inputs_pp + models_optimization_pp + data_science_pp
        return pipelines

    return register_pipelines