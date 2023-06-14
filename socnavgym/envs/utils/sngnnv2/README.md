# Repository information
This document explains the structure of this repository and how to use it to train and test three different GNN models that you can find in the `nets` directory. 
This repository reproduces the experiments conducted in the paper: https://arxiv.org/pdf/2102.08863.pdf

## Using the model
The script 'single_test.py' demonstrates how to use the API:
```
$ git clone https://github.com/gnns4hri/sngnnv2
$ cd sngnnv2/
$ python3 single_test.py example_model/ jsons_test/S1_000003.json 
Result for a *single* query: tensor([0.9363, 0.7759]). It took 0.21820592880249023 seconds
```
Bear in mind that the script **does not use CUDA by default**.

## Download the dataset
The first step to train a model is to get the data.
You can download our dataset from [here](https://gnns4hri.org/data.zip). 

The downloaded file must be unzipped in the `raw_data` directory under the directory `data`.
It contains several JSON files with the information of the scenarios and its labels. 
Hence, each file stores data for several consecutive frames of the simulation which correspond to one label.
In the end, you will have the following path to the JSON files:

```bash
./raw_data/data
```
Notice that there are 3 txt files in the raw data directory. 
These are used by the training script to divide the dataset into train, dev and test sets.

## Training instructions
Once the dataset is set up, we can train our models with that data. 
We provide two ways of performing the training process: setting the hyperparameters manually to train a single model or perform several trainings with random hyperparameters.

### Single model training
This section explains how to perform the training of a single model imputing the hyperparameters by hand.
For doing that, it is necessary to edit the script `train.py` in the lines 316 to 335.
Next, you can find an example:

```python
best_loss = main('train_set.txt', 'dev_set.txt', 'test_set.txt',
                graph_type='1',
                net='mpnn',
                epochs=1000,
                patience=6,
                batch_size=31,
                num_classes=2,
                num_hidden=[25, 20, 20, 15, 10, 3],
                heads=[15, 15, 10, 8, 8, 6],
                residual=False,
                lr=0.0005,
                weight_decay=1.e-11,
                nonlinearity='elu',
                final_activation='relu',
                gnn_layers=7,
                in_drop=0.,
                alpha=0.28929123386192357,
                attn_drop=0.,
                cuda=True,
                fw='dgl')
```
`graph_type` must be always set to 1 and `fw` to dgl.
For the `net` field you can choose among the three kinds of models that are currently implemented:
"mpnn" for Message Passing Neural Network.
"gat" for Graph Attention Network.
"rgcn" for Relational Graph Convolutional Network.
`num_hidden` field indicates the input hidden features for each of the layers of the model and must match with the `gnn_layers` + 1 field.
`heads` filed set the dimension of the heads in the head attention mechanism implemented by the GAT model.
The rest of the parameters are self-explanatory.

Once all the hyperparameters are set, the last step is to run the script:
```bash
python3 train.py
```

When it finishes, it will create a directory called "model_params" that store two files with the parameters and weights of the trained model.

### Batched training
This way of training will launch a sequence of training processes with random hyperparameters.
The first step is to create the list of task running the following script:

```bash
python3 generate_training_hyperparameter_samples.py
```
It will create a list of tasks stored in the pickle file "LIST_OF_TASKS.pckl" where each task is a set of hyperparameters for a training.

The next step is to start the training process with the following command:

```bash
python3 run_script.py "python3 train_batched.py"
```
It will train until the script is stopped. 
Each model trained will be stored in a directory with an ID number as a name.
If you run ´generate_training_hyperparameter_samples.py´ script again it will print the ID of the model with the best loss.

## Testing instructions
This repository has a script (`showcase.py`) to generate a heat-map with one of the trained models.
It must be called using the following format:

```bash
python3 showcase.py "example_model" "file.json" resolution
```
The first argument must be the name of the directory where the model params are stored.
The second argument is the name of the JSON file that must be inside jsons_test directory.
Finally, the third argument is the desired resolution of the output image. 
Note that the higher the resolution the slower the creation process.

In this repository, we provide a trained model as an example called "example_model".
Besides, there is a directory ("jsons_test") that has JSON files for testing three different scenarios.
It also has videos of these scenarios.

