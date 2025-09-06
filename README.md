# Optimizing Classification Performance in Google Earth Engine 
## Background
In recent years, cloud-based platforms such as Google Earth Engine (GEE) have become increasingly popular across a wide range of scientific and applied disciplines. Providing easy access to petabytes of Earth observation data, GEE has transformed remote sensing workflows, making it possible to perform large-scale and long-term analyses that were once constrained by local computing power. GEE supports a wide range of operations, from basic spatial and spectral calculations to more advanced machine learning tasks ([Perez-Cutillas et al 2023](https://doi.org/10.1016/j.rsase.2022.100907)). Both supervised and unsupervised classification are available in GEE, with popular algorithms like Random Forest (RF), Support Vector Machine (SVM), and Classification and Regression Trees (CART) ready to use. However, one key limitation of GEE is the lack of built-in hyperparameter optimization, a feature commonly found in libraries such as scikit-learn. Since the performance and generalization of machine learning models often depend on the choice of parameters, tuning them is a critical step in any classification workflow.
This repository explores practical approaches for improving classification performance in the GEE Python API. The methods include strategies for optimal data partitioning (e.g., stratified train-test splits) and grid search–like parameter tuning, adapted for use in Earth Engine workflows.
## Data Partitioning
The first steps in machine learning workflow is partioned your original training data/samples into training and testing data. This partitioning is important since one of the advantages in machine learning workflow is the ability to asses if your model is able to captured the pattern in the data. Many example online of data partitioning is primarily focus on single random split, which blindy partitioned the training data based on certain split ratio. This approach can be suitable if you have equal amount of samples across all class. However, one of the common trends in land cover phenomena is certain land cover types often dominate other. As the result, imbalance data were prominent in land cover mapping task. Single random split may cause underrepresented data to be overlooked and severely imbalanced. To mitigate this effect, stratified random split is implemented here, which used the class strata to perserve the original proportion of the data. Below are the functions to conduct stratified train-test split in python API
  1. first, we aggregate the distinct class property ('class_prop') of the training data to get unique class id 
```python
classes = roi.aggregate_array(class_prop).distinct()
```
  2. Then the region of interest (roi) is split based on:
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
Then we map the function (split_class) 
```python
#map the function for all the class
split_fc = ee.FeatureCollection(classes.map(split_class)).flatten()
#filter for training and testing
train_fc = split_fc.filter(ee.Filter.eq('fraction', 'training'))
test_fc = split_fc.filter(ee.Filter.eq('fraction', 'testing'))
```


