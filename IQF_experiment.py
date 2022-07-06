
import os

from iquaflow.datasets import DSModifier, DSWrapper,DSModifier_jpg
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.experiment_visual import ExperimentVisual
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.metrics import BBDetectionMetrics

#Define name of IQF experiment
experiment_name = "experimentA"

#Define path of the original(reference) dataset
data_path = "dataset/SateAirports"

#Define path of the training script
python_ml_script_path = "./iqt_train_wrapper.py"

#DS wrapper is the class that encapsulate a dataset
ds_wrapper = DSWrapper(data_path=data_path)

#List of modifications that will be applyed to the original dataset:
# DSModifier refers to a modifier that simply does nothing - Reference training
# DSModifier_jpg apply JPG compresion to the dataset

ds_modifiers_list = [ DSModifier_jpg(params={'quality': i}) for i in [10,30,50,70,90] ]

# Task execution executes the training loop
task = PythonScriptTaskExecution(model_script_path=python_ml_script_path)

#Experiment definition, pass as arguments all the components defined beforehand
experiment = ExperimentSetup(
    experiment_name=experiment_name,
    task_instance=task,
    ref_dsw_train=ds_wrapper,
    ds_modifiers_list=ds_modifiers_list,
    repetitions=5,
    extra_train_params={'min_size':[11,16,20,30]}
)

#Execute the experiment
experiment.execute()

# ExperimentInfo is used to retrieve all the information of the whole experiment. 
# It contains built in operations but also it can be used to retrieve raw data for futher analysis

experiment_info = ExperimentInfo(experiment_name)

experiment_info.apply_metric_per_run(
    BBDetectionMetrics(),
    ds_wrapper.json_annotations
)

runs = experiment_info.runs
