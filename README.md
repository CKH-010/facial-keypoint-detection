# COMP 4330SEF - Facial Keypoint Detection with Transfer Learning (P06 Group 5)

The Udacity dataset annotations use the original image's pixel coordinates;
source image dimensions vary. The dataset pipeline resizes images to 96 x 96
and normalizes each target axis by the corresponding original image dimension.
Multiply predictions by 96 to display them on a 96 x 96 image. The model's
outputs are unconstrained, so predictions are not guaranteed to fall within
the image.

After changing the target scaling, regenerate the cached `data/all_*.pt` files
and retrain the regression head; checkpoints trained with the old scaling
retain the old coordinate convention. The checked-in training notebook trains
the baseline from scratch and does not contain an advanced-model transfer
training path. Baseline convolutional weights may still be useful for transfer,
but the first convolution needs adaptation for CoordConv's two extra channels,
and an old regression head should not be reused without retraining.

`Aligned_Training.ipynb` trains both models directly from the CSV and source
images instead of loading potentially stale caches. Run all its cells in order;
it saves new `*_aligned.pth` checkpoints and plots matching ground truth and
predictions for valid keypoints only. If running it in another project folder,
copy the notebook, `src/data_pipeline.py`, and `src/alignment_workflow.py` into
that folder first. Existing caches and checkpoints are not updated by it.
