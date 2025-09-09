# Optimizing Classification Performance in Google Earth Engine 
## Background
In recent years, cloud-based platforms such as Google Earth Engine (GEE) have become increasingly popular across a wide range of scientific and applied disciplines. Providing easy access to petabytes of Earth observation data, GEE has transformed remote sensing workflows, making it possible to perform large-scale and long-term analyses that were once constrained by local computing power. GEE supports a wide range of operations, from basic spatial and spectral calculations to more advanced machine learning tasks ([Perez-Cutillas et al 2023](https://doi.org/10.1016/j.rsase.2022.100907)). Both supervised and unsupervised classification are available in GEE, with popular algorithms like Random Forest (RF), Support Vector Machine (SVM), and Classification and Regression Trees (CART) ready to use. However, one key limitation of GEE is the lack of built-in hyperparameter optimization, a feature commonly found in libraries such as scikit-learn. Since the performance and generalization of machine learning models often depend on the choice of parameters, tuning them is a critical step in any classification workflow.
This repository explores practical approaches for improving classification performance in the GEE Python API. The methods include strategies for optimal data partitioning (e.g., stratified train-test splits) and grid search–like parameter tuning, adapted for use in Earth Engine workflows.
## Data Partitioning
The first steps in machine learning workflow is partioned your original region of interest (ROI) into training and testing data. This partitioning is important since one of the advantages in machine learning workflow is the ability to asses if your model is able to captured the pattern in the data. Many example online of data partitioning is primarily focus on single random split, which blindy partitioned the training data based on certain split ratio. This approach can be suitable if you have equal amount of samples across all class. However, one of the common trends in land cover phenomena is certain land cover types often dominate the other, resulting in imbalance dataset. Direct random split may cause some class to be severely underepresented and thus underfit the model. To mitigate this effect, stratified random split is implemented here, which used the class strata to perserve the original proportion of the data. Although some studies have caution the single split approach as aspatial and did not consider the spatial autocorrelation effect in the region of interest ([Ramezan et al 2023](https://doi.org/10.3390/rs11020185)). Since spatial dependent train test split is complicated, stratified random split is implemented here. You can find the complete functions in feature_extraction.py file. Here i am going to walk you through the process
For context, the stratified split functions required the following parameters:
 - roi: ee.FeatureCollection of training geometries with a class label
 - image: ee.Image whose bands you want to sample.
 - class_prop:  name of the label property in ROI (e.g., "class").
 - pixel_size: sampling scale (e.g., 10 m for Ssenitinel 2, 30 m for Landsat).
 - train_ratio: proportion per class for training (e.g., 0.7).

seed — random seed for reproducible splits.
  1. first, we aggregate the distinct class property ('class_prop') of the training data to get unique class id. The result of this process is a list contain the unique ID of the class
```python
classes = roi.aggregate_array(class_prop).distinct()
```
  2. Now we define per-class splitter, which start by filtering the ROI to one class. Then, add uniform random number ('random' in [0,1)) to each feature using a fixed seed. The split ratio is used as mark the feature between training and testing. Finnally, merge the two features for a single class
```python
  def split_class (c):
            subset = (roi.filter(ee.Filter.eq(class_prop, c))
                    .randomColumn('random', seed=seed))
            train = (subset.filter(ee.Filter.lt('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'training')))
            test = (subset.filter(ee.Filter.gte('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'testing')))
            return train.merge(test)
```
  3. Then we map the sub-function (split class) into the whole featureCollection with a 'fraction' property on each feature.
```python
#map the function for all the class
split_fc = ee.FeatureCollection(classes.map(split_class)).flatten()
#filter for training and testing
train_fc = split_fc.filter(ee.Filter.eq('fraction', 'training'))
test_fc = split_fc.filter(ee.Filter.eq('fraction', 'testing'))
```
  4. The final steps is extract the pixel value from the imagery using the SampleRegion codes
```python
train_pix = image.sampleRegions(
                            collection=train_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16)
test_pix = image.sampleRegions(
                            collection = test_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16)
``` 

In addition to stratified random split, i also documented the stratified k-fold split, which partitioned the roi into a specified folds/subset. This approach is more robust for model learning and evaluation, however required a significant exhaustive computation time. The approach i used is adapted from [spatial though courses](https://courses.spatialthoughts.com/end-to-end-gee-supplement.html#k-fold-cross-validation)

## Classification Approach
In land cover mapping task, machine learning classifiers are typically applied in a multiclass (or "hard") classification setting. In this approach, the algorithm directly assigns each pixel to one of the available land cover classes. While this is the most straight forward method, this approach often underperform to classified the underepresented class, especially if there's some land cover type which dominate the area of interest. The one-vs-rest classification (OVR) serve as an alterntive approach by decomposing multiclass problems into set of binary classification task for each class. For every pixel, the algorithm evaluates whether it belongs to the target class (one) or to any of the other classes (rest). This process produces a set of probability layers, where values closer to 1 indicate higher confidence that a pixel belongs to a given class. To generate the final land cover map, these probability layers is combined and each pixels is assigned to the class with highest probability score. In addtion to maximum probability, assembly of the probability layers can be conducted using various approach to give more control to the end result. If your are interested in this topic please visit [RESTORE+ Project](https://www.restoreplus.org/uploads/1/0/4/5/104525257/restore__technical_report_land_cover_mapping_july2022.pdf) and [Saah et al 2020](https://doi.org/10.1016/j.jag.2019.101979).

## Parameter Optimization
From the classification approach, we can decide which parameter optimization suitable for the mapping task. In the python file, i have provide 4 approach for parameter optimization:
 - 
Since GEE did not have built-in hyperparameter, we have to manually test the parameter combinations, evaluate them, and rank its accuracy. This approach is similar to grid search approach in scikit-learn library, which characterize as having exhaustive time with more parameters and combination tested. In the classifier_tuning.py file, you can fine 4 tuning options. Two 

