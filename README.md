# Densenet_Tenserflow
This repository is a densenet implementation writing by tensorflow. In this implementation, you don't need to write code to build the network, which you need is to write config file about the network you want to run.

In machine learning, the main phases of building a training network contains preparing dataset, building network and choosing training parameters. So in this implementation, you only need to modify the *xx_input.config, train_parameter.config* and *densenet_xx.config* files in *config_files* folder. The meaning of each key words in config files is defnind in proto files in path *protos*.

This implementation only support tfrecords input. The code about transforming from original images to tfrecords format are in *datasets* folder.

You should modify the following place to make the code work.
1. In *xx_input.config* file, changing the **file_location** to your location: (*cifar10_input_dataset.config* file, line 6)
 ```
 file_location: '/home/aurora/workspaces/data/tfrecords_data/cifar_dataset'
 ```

 2. In *train_parameter.config*, changing the **densenet_config_path** to the densenet architecture you want to use.
 ```
 densenet_config_path: 'densenet/densenet_cifar10_bc.config'
 ```
 I have write some densenet config files in *config_files/densenet* folder.

### train the model
```
cd train_network
python densenet_cifar10_train_dataset_api.py
```

### Dependencies
* Tensorflow 1.3
* Protobuf

### Some Results
| network  |  dataset   | accuracy  |   weight deacy |
| :------------: | :------------: | :------------: | :------------: |
|  densenet(L=40, K=12) |     cifar10    |    0.91    |  no    |
|  densenet(L=40, K=12) |     cifar10    |    0.92    |  yes |

### Reference
[DenseNet](https://github.com/liuzhuang13/DenseNet)
[vision_networks](https://github.com/ikhlestov/vision_networks)
