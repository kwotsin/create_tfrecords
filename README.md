# create_tfrecords
A simpler way of preparing large-scale image dataset by generalizing functions from TensorFlow-slim.

### Requirements
1. Python 2.7.x
2. TensorFlow >= 0.12

**NOTE**: If you want to run this program on Python 3, clone and run `git checkout python-3.0` for the Python 3 branch instead.

### Usage

    $python create_tfrecord.py --dataset_dir=/path/to/dataset/ --tfrecord_filename=dataset_name
    
    #Example: python create_tfrecord.py --dataset_dir=/path/to/flowers --tfrecord_filename=flowers
    #Note that the dataset_dir should be the folder that contains the root directory and not the root directory itself.

### Arguments

#### Required arguments:

- dataset_dir (string): The directory to your dataset that is arranged in a structured way where your subdirectories keep classes of your images. 

For example:

    flowers\
        flower_photos\
            tulips\
                ....jpg
                ....jpg
                ....jpg
            sunflowers\
                ....jpg
            roses\
                ....jpg
            dandelion\
                ....jpg
            daisy\
                ....jpg
   
  Note: Your dataset_dir should be /path/to/flowers and not /path/to/flowers/flowers_photos

- tfrecord_filename (string): The output name of your TFRecord files.

#### Optional Arguments
- validation_size (float): The proportion of the dataset to be used for evaluation.

- num_shards (int): The number of shards to split your TFRecord files into.

- random_seed (int): The random seed number for repeatability.

### Complete Guide
For a complete guide, please visit [here](https://kwotsin.github.io/tech/2017/01/29/tfrecords.html).
