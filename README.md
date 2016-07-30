###################################################################
#                                                                 #
#                  Symmetry-based text detection                  #
#                                                                 #
#                                                                 #
###################################################################

1. Introduction.

The source code of symmetry-based text detection algorithm. We adopt approximate calculation technology and parallel technology to speed up the proposed algorithm.
For this reason, the probability output has a slightly difference but very close result on ICDAR dataset, compared to the CVPR'15 version.

###################################################################

2. Installation.

a) This code is written for the Windows x64 and Visual Studio 2012, and you may needs OpenCV 2.4.10 and VL-feat. 

b) You should check the config.txt for parameter setting.
   The config.txt is consists of several lines:
   line #1: Dataset Path
   line #2: Mode flag(0 or 1), 0 is testing mode and 1 is training mode
   line #3: The normalized height of input image
   line #4: number of sliding window scales for each octave
   line #5: the minimum sliding window scale, it would be 2^min_scale
   line #6: the maximum sliding window scale, it would be 2^min_scale
   line #7: Only for training stage. This parameter indicate an absolute distance. 
	    Pixels whose distance greater than this distance are regard as negative samples.
   line #8: Only for training stage. This parameter indicate an relative distance(relative to the height of ground truth, Eg: if the height of ground truth is 10 and the paprameter is 0.2, the distance is 2). 
	    Pixels whose distance less than this distance are regard as positive samples
   line #9: Only for training stage. The maximum negative samples in training stage.
   line #10: Only for training stage. The maximum positive samples in training stage.
   line #11: the bin number of lab channel
   line #12: the bin number of gradient channel
   line #13: the bin number of texture channel. In current version, this parameter is fixed by 58
   line #14: the angle threshold for symmetry line linkage
   line #15: the core number for parallel
   line #16: the shrink step, it should be 2^n

   and We support an example in the package. Note that the setting of config.txt(line #11 ~ line#13) should be keep consistent during training and testing.

c) Prepare the dataset and create folders.
   The directory structure is described as bellow:
   ->Dataset\
	->Annotation\
		->train\
	->Feature
		->train\
	->Images
		->train\
		->test\
	->Model
	->Result
	
    And for annotation files, each of them should contains several lines, and each line should contain 4 numbers: x, y, width, height.
    For more details, please refer to the sample in the package.
	
###################################################################

3. Getting Started.

 - Make sure to carefully follow the installation instructions above.
 - Set config.txt.
 - Run TextlineDetection.exe config.txt in command line.

###################################################################
