# VPC-Net: Completion of 3D Vehicles from MLS Point Clouds
[VPC-Net: Completion of 3D Vehicles from MLS Point Clouds](https://www.sciencedirect.com/science/article/abs/pii/S0924271621000344)  ISPRS Journal (IF = 7.319) 

Yan Xia, Yusheng Xu, Cheng Wang, Uwe Stilla

Technical University of Munich, Xiamen University

## Introduction

VPC-Net is a neural network to synthesize complete, dense, and uniform point clouds for vehicles from MLS data. The arXiv version of VPC-Net can be found [here](https://arxiv.org/abs/2008.03404).

![xia-VPCNet-Real-time-vehicle-com](./demo/xia_VPCNet_Real-time_vehicle_completion_using_Kitti_dataset.gif)

### Citation

> ```
> @article{xia2021vpc,
>   title={VPC-Net: Completion of 3D vehicles from MLS point clouds},
>   author={Xia, Yan and Xu, Yusheng and Wang, Cheng and Stilla, Uwe},
>   journal={ISPRS Journal of Photogrammetry and Remote Sensing},
>   volume={174},
>   pages={166--181},
>   year={2021},
>   publisher={Elsevier}
> }
> ```

This code is built using Tensorflow 1.12 with CUDA 9.0 and tested on Ubuntu 16.04 with Python 3.5.

### Complie TF Operators

------

Please follow [PointNet++](https://github.com/charlesq34/pointnet2) to compile TF operators.

### Training

------

1. Download `shapenet_car` directory from Google Drive. 
2. Run `python train.py`

### Testing

------

1. ShapenNet-car completion
   - run `python test.py`.
2. KITTI completion
   - Download KITTI data from [PCN](https://github.com/wentaoyuan/pcn) project.
   - run `python test_kitti.py`.

### Acknowledgements

Our implementations based on [PCN](https://github.com/wentaoyuan/pcn) repository.