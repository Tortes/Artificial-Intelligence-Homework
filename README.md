# Product defect detection
## Introduction
Detect the defect of products using deep neural network
## Requirements
- Python 3.x
- tensorflow 1.8.0
- opencv 4.0
## Usage
### For output the result with pre-trained network
- Copy the test-needed images to `./data/dataset/origin`
- Run the  `./scripts/output.py` to get the result, the file `result.txt` is placed at the root directory 
- Run the  `./scripts/outputvisual.py` to get the visual result with dividing the image with three parts, which is labeled with `pass`, `slant`(焊偏), `black`(焊黑)

### For train the network with your own images
- Copy the labeled images to `./data/dataset/pass` and `./data/dataset/fail`
- Run the `./scripts/dataprocess` to generate the required files
- Run and set the parameter in  `./scripts/train.py` to train your own images 