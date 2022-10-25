# *This use case is under construction. The dataset is not available yet.*


# Airplane Detection IQF Experiment

Airplaines can be detected from satellite images by AI algorithms. The objective of this study is to analize the effect of image compression on the detection performace.

The core training code is an adaptation from [source repo](https://github.com/jwyang/faster-rcnn.pytorch)

To make it compatible to your GPU hardware:
 - See comments [here](https://github.com/jwyang/faster-rcnn.pytorch#compilation) about your GPU hardware.
 - Setup cuda version in the pytorch=0.4.0 installation within the Dockerfile

To reproduce the experiments:

1. `git clone git@publicgitlab.satellogic.com:iqf/iquaflow-airport-use-case.git`
2. `cd iquaflow-airport-use-case`
3. Then build the docker image with `make build`. This will also download the dataset in `./dataset` as well as these weights:
    ```
    ./data/pretrained_model/resnet101_caffe.pth
    ./data/cache/sate_airports_trainval_gt_roidb.pkl
    ./data/cache/sate_airports_trainval_sizes.pkl
    ./models/res101/sate_airports/faster_rcnn_1_7_10021.pth
    ```
4. In order to execute the experiments:
    - `make dockershell` (\*)
    - Inside the docker terminal execute `python ./iqf_experiment.py`
5. Start the mlflow server by doing `make mlflow` (\*)
6. Notebook examples can be launched and executed by `make notebookshell NB_PORT=[your_port]"` (\**)
7. To access the notebook from your browser in your local machine you can do:
    - If the executions are launched in a server, make a tunnel from your local machine. `ssh -N -f -L localhost:[your_port]:localhost:[your_port] [remote_user]@[remote_ip]`  Otherwise skip this step.
    - Then, in your browser, access: `localhost:[your_port]/?token=AIRPORT`


____________________________________________________________________________________________________

    - The results of the IQF experiment can be seen in the MLflow user interface.
    - For more information please check the IQF_expriment.ipynb or IQF_experiment.py.
    - There are also examples of dataset Sanity check and Stats in SateAirportsStats.ipynb
    - The default ports are 8888 for the notebookshell, 5000 for the mlflow and 9197 for the dockershell
    - (*)
        Additional optional arguments can be added. The dataset location is:
            DS_VOLUME=[path_to_your_dataset]`
    - To change the default port for the mlflow service:
            `MLF_PORT=[your_port]`
    - (**)
        To change the default port for the notebook:
            `NB_PORT=[your_port]`
    - A terminal can also be launched by `make dockershell` with optional arguments such as (*)

