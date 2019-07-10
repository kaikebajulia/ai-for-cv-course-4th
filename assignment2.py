# Assignment Week 2:
'''
You needn't finish reading all of them in just one week!
It's just good for you to know what's happening in this area and to figure out how people try to improve SIFT.

You needn't to remember all of them. 
But please DO REMEMBER procedures of SIFT and HoG. For those who're interested in SLAM, Orb is your inevitable destiny.

[Reading]:
1. [optional] Bilateral Filter: https://blog.csdn.net/piaoxuezhong/article/details/78302920
2. Feature Descriptors:
   [Compulsory]
   Hog: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
   SURF: https://www.vision.ee.ethz.ch/~surf/eccv06.pdf
   [optional]
   BRISK: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.1343&rep=rep1&type=pdf
   Orb: http://www.willowgarage.com/sites/default/files/orb_final.pdf [Compulsory for SLAM Guys]
3. Preview parts:
   K-Means: I have no doubts about what you are going to read and where you gonna find the reading materials. 
            There are tons of papers/blogs describing k-means. Just grab one and read.
			We'll this topic in 3 weeks.
			
[Coding]:			
1. 
#    Finish 2D convolution/filtering by your self. 
#    What you are supposed to do can be described as "median blur", which means by using a sliding window 
#    on an image, your task is not going to do a normal convolution, but to find the median value within 
#    that crop.
#
#    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When 
#    "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis 
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version 
#    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#    Follow up 1: Can it be completed in a shorter time complexity?
#
#    Python version:
#    def medianBlur(img, kernel, padding_way):
#        img & kernel is List of List; padding_way a string
#        Please finish your code under this blank
#
#
//   C++ version:
//   void medianBlur(vector<vector<int>>& img, vector<vector<int>> kernel, string padding_way){
//       Please finish your code within this blank  
//   }

2. 【Reading + Pseudo Code】
#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A, 
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding 
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like: 
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#       
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while 
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold 
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#
#       Python:
#       def ransacMatching(A, B):
#           A & B: List of List
#
//      C++:
//      vector<vector<float>> ransacMatching(vector<vector<float>> A, vector<vector<float>> B) {
//      }    
#
#       Follow up 1. For step 3. How to do the "test“? Please clarify this in your code/pseudo code
#       Follow up 2. How do decide the "k" mentioned in step 5. Think about it mathematically!
#
# You are supposed to hand in the code in 1 week.
#


[Classical Project]
1. Classical image stitching!
   We've discussed in the class. Follow the instructions shown in the slides.
   Your inputs are two images. Your output is supposed to be a stitched image.
   You are encouraged to follow others' codes. But try not to just copy, but to study!
   
   You are supposed to hand in this project in 2-3 weeks.
			
'''
      

