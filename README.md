# UnetPointAnnotationSegmentation

# Model Result Kaggle: https://www.kaggle.com/parkwonro/pointannotation-and-segmentation-model-result

# Used Dataset
DAVIS: A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation (DAVIS) https://davischallenge.org/
============================================================================================

Documentation
-----------------

The directory is structured as follows:

 * `ROOT/JPEGImages`: Set of input video sequences provided in the form of JPEG images.
   Video sequences are available at 1080p and 480p resolution.

 * `ROOT/Annotations`: Set of manually annotated binary images providing reference segmentation
   for the foreground objects. Annotations are available at 480p resolution and Point data.

 * `ROOT/Annotations/Point`: Set of applying a Gaussian Filter(shape(7*7), segma:2.5) about random pixel(255) in mask
