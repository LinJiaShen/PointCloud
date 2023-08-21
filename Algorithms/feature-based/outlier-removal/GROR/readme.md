
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
For example ./GrorReg data/src.pcd data/tgt.pcd 0.1 800 data/src_cor.txt data/src_tgt.txt


# Debug Log:
**1. Boost error:**<br>
In the original main.cpp have a unsolvable error, which caused in the last function 
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
The issue arises due to a different version of C++ (boost); however, using other versions does not resolve the problem either.
