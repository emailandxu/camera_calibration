# camera_calibration

This is a visualization tool for SfM toolkits including openMVG and Colmap.
And a dataset_loader implentation for training gaussian splatting scene.


## Visualization
build docker by openmvg offcial build [guide](https://github.com/openMVG/openMVG/blob/085fbe4f740b31c8a0ae5b824451eae68199ea63/BUILD.md?plain=1#L184).

then, run commands below to make a 360 camera video into a 360 camera dataset that contains cubic images in *images* subfolder, point clound in *reconstruction_global/colorized.ply* and camera calibration in *sfm_data_perspective.json*.

```bash
ffmpeg -i VID_20231023_123406_00_034.mp4 -vf "fps=1" input/%04d.jpg
docker run --rm -it -v $PWD:$PWD openmvg python3 $PWD/scripts/my_sfm_sphere.py $PWD/db/restroom/input $PWD/db/restroom/output
sudo chown -R $(whoami) $PWD/db/restroom/output
```

### openmvg
```bash
python main.py \
--dataset-type openmvg \
--camera-meta db/restroom/output/sfm_data_perspective.json \
--image-folder db/restroom/output/images \
--plypath db/restroom/output/reconstruction_global/colorized.ply \
--scale 0.01
```

### colmap
```bash
python main.py \
--dataset-type colmap \
--camera-meta db/avenue-cubic/sparse/0/images.txt \
--image-folder db/avenue-cubic/images \
--plypath db/avenue-cubic/sparse/0/points3D.ply \
--scale 0.05
```

## Training Gaussian splatting
Clone Gaussian Splatting offcial repository, and copy paste scripts in 3rdpart/scene to it.
Then train the scene accordding to offcial guide.
