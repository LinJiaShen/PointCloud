
# Install GROR<br>

Dependencies:
---
1. OpenMP  [install guide](https://medium.com/swlh/openmp-on-ubuntu-1145355eeb2)
2. Cmake >= 3.5
3. PCL >= 1.8.1

Compilation:
---
```
git clone https://github.com/WPC-WHU/GROR.git
cd GROR; mkdir build; cd build
cmake ..
make
```
Before make, the main.cpp from git clone should be replaced to our version main.cpp. Due to the original design of GROR adopt the strategy of (downsampling) + (SIFT key point extraction) + （FPFH descriptor) + (correspondences matching), which we do not need to use in this study, we aim to input the correspondences from different feature extraction algorithms.

After replacing the main.cpp, run `make` in the build folder. Then, there will be a GrorReg that can execute the algorithm.

Usage:
---
```
./GrorReg "source path" "target path" resolution n_optimal "source correspondence path" "target correspondence path"
```
For example <br>
./GrorReg data/src.pcd data/tgt.pcd 0.1 800 data/src_cor.txt data/src_tgt.txt


# Debug Log:
**1. Boost error:**<br>
In the original main.cpp, there is an unsolvable error that occurs in the last function.
```
auto ShowVGFPointCloud2 = [](pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT)
{
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorT(cloudT, 0, 100, 160);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorS(cloudS, 255, 85, 0);
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerVGF(new pcl::visualization::PCLVisualizer("After Registration"));
		viewerVGF->setBackgroundColor(255, 255, 255);
		viewerVGF->addPointCloud<pcl::PointXYZ>(cloudS, colorS, "source cloud");
		viewerVGF->addPointCloud<pcl::PointXYZ>(cloudT, colorT, "target cloud");
		viewerVGF->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");
		viewerVGF->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
		while (!viewerVGF->wasStopped())
		{
			viewerVGF->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));    <- the error caused from here
		}
	};
```
The issue arises due to a different version of C++ (boost); however, using other versions does not resolve the problem either.<br>
**2. AttributeError: module 'numpy' has no attribute 'bool'.**<br>
In the ./lib/bechmark.utils.py `matual_selection` function:
replace the np.bool to bool
```
def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column
    
    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N] 
    """
    score_mat=to_array(score_mat)
    if(score_mat.ndim==2):
        score_mat=score_mat[None,:,:]
    
    mutuals=np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]): # loop through the batch
        c_mat=score_mat[i]
        flag_row=np.zeros_like(c_mat)
        flag_column=np.zeros_like(c_mat)

        max_along_row=np.argmax(c_mat,1)[:,None]
        max_along_column=np.argmax(c_mat,0)[None,:]
        np.put_along_axis(flag_row,max_along_row,1,1)
        np.put_along_axis(flag_column,max_along_column,1,0)
        mutuals[i]=(flag_row.astype(bool)) & (flag_column.astype(bool))
    return mutuals.astype(bool) 
```
