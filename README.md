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
From the classification approach, we can decide which parameter optimization is suitable for the mapping task. In the python file, i have provided 4 approaches for parameter optimization in classifier_tuning.py:
- Hard Classification tuning: This function is used if the classification approach is hard multiclass classification
- Soft classification tuning: This function is used if the classification approach is One-vs-Rest Binary classification framework
- Hard fold classification tuning: This function is used for multi-class classification with k-fold data
- Soft fold classification tuning: This function is used if the classification approach is One-vs-Rest Binary classification framework with kfold data
Since GEE did not have built-in hyperparameter optimization, we must manually test the parameter combinations, evaluate them, and rank their accuracy. This task can be conducted using looping strategies in python. This approach is akin to grid search hyperparameter tuning in scikit-learn library, characterized by having exhaustive time with more parameters and combination tested. In this example, I am using random forest classifiers, since they are one of the most widely used machine learning algorithms in remote sensing study. To avoid user memory limitations, we only test 3 main parameters, namely Number of trees (n_tree), number of variables selected at split (var_split), and minimum sample population at leaf node (min_leaf_pop). If you’re working with large area of interest or detailed classification schemes, I highly recommend tuning only two parameters, n_tree and var_split, since Maxwell et al 2018 stated that these two parameters is one of the most significant parameters.

 In the functions, several input were required to execute the functions. The primary inputs were training pixels (train), testing pixels (test), image, land cover id (class_property), and parameter list.

The functions start with creating and empty list to stored the parameter combinations and resulting accuracy. Additionally, the following codes are used to calculate how many parameters combinations
```python
        result = [] #initialize empty dictionary for storing parameters and accuracy score
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting hyperparameter tuning with {total_combinations} parameter combinations...")
```

Below are the main looping strategies in which each parameters combination is tested and evaluated. Each resulting accuracy is then stored in previously defined list. The progress of the tuning is printed using the tqdm library for easier monitoring. 
```python
        with tqdm(total=total_combinations, desc="Hard Classification Tuning") as pbar:
            for n_tree in n_tree_list:
                for var_split in var_split_list:
                    for min_leaf_pop in min_leaf_pop_list:
                        try:
                            print(f"Testing: Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                            #initialize the random forest classifer
                            clf = ee.Classifier.smileRandomForest(
                                numberOfTrees=n_tree,
                                variablesPerSplit=var_split,
                                minLeafPopulation = min_leaf_pop,
                                seed=0
                            ).train(
                                features=train,
                                classProperty=class_property,
                                inputProperties=image.bandNames()
                            )
                            #Used partitioned test data, to evaluate the trained model
                            classified_test = test.classify(clf)
                            #test using error matrix
                            error_matrix = classified_test.errorMatrix(class_property, 'classification')
                            #append the result of the test
                            accuracy = error_matrix.accuracy().getInfo()
                            result.append({
                                'numberOfTrees': n_tree,
                                'variablesPerSplit': var_split,
                                'MinimumleafPopulation':min_leaf_pop,
                                'accuracy': accuracy
                            })
                            #print the message if error occur
                        except Exception as e:
                            print(f"Failed for Trees={n_tree}, Variable Split={var_split}, mininum leaf population = {min_leaf_pop}")
                            print(f"Error: {e}")
                            
                        finally:
                            pbar.update(1)

```
Finally, the list which contains the parameters combinations and its accuracy is converted to panda dataframe for easier interpretation. 
```python
            if result:
                result_df = pd.DataFrame(result)
                result_df_sorted = result_df.sort_values(by='accuracy', ascending=False)#.reset_index(drop=True)
                
                print("\n" + "="*50)
                print("GRID SEARCH RESULTS")
                print("="*50)
                print("\nBest parameters (highest model accuracy):")
                print(result_df_sorted.iloc[0])
                print("\nTop 5 parameter combinations:")
                print(result_df_sorted.head())
                
                return result, result_df_sorted
            else:
                print("No successful parameter combinations found!")
                return [], pd.DataFrame()
```
For the soft (binary) classification one of the main difference is how to evaluate the model. For this approach, i used cross-entropy loss metric indicating how confidence the model in producing the probability result. Additionally, this metric is recommended by [Farhadpour et al 2024](https://doi.org/10.3390/rs16030533) for model evaluation since it is insensitive to imbalance dataset. 
The overall function structure is the same, with some difference in the looping strategy and classification
```python
        with tqdm(total=total_combinations, desc="Hard Classification Tuning") as pbar:  
            for n_tree in n_tree_list:
                for var_split in var_split_list:
                    for min_leaf_pop in min_leaf_pop_list: 
                        try:
                            print(f"Testing: Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                            def per_class(class_id):
                                class_id = ee.Number(class_id)

                                binary_train = train.map(lambda ft: ft.set(
                                    'binary', ee.Algorithms.If(
                                        ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                                    )
                                ))
                                binary_test = test.map(lambda ft: ft.set(
                                    'binary', ee.Algorithms.If(
                                        ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                                    )
                                ))
                                #Random Forest Model, set to probability mode
                                clf = (ee.Classifier.smileRandomForest(
                                        numberOfTrees = n_tree,
                                        variablesPerSplit = var_split,
                                        minLeafPopulation = min_leaf_pop,
                                        seed = seed)
                                        .setOutputMode('PROBABILITY'))
                                model = clf.train(
                                    features = binary_train,
                                    classProperty = 'binary',
                                    inputProperties = image.bandNames()
                                )
                                test_classified = binary_test.classify(model)
```
As we can see above, prior to the model training, we convert the training and testing data into a binary class, with 1 as true pixels, and 0 as false pixels. Furthermore, the rendom forest output mode is set to 'PROBABILITY' which resulted in 1 probability layer for each class.  

The following codes is used to calculate the cross entropy loss (log loss)
```python
                                # Extract true class labels and predicted probabilities
                                y_true =  test_classified.aggregate_array('binary')
                                y_pred =  test_classified.aggregate_array('classification')
                                paired = y_true.zip(y_pred).map(
                                        lambda xy: ee.Dictionary({
                                            'y_true': ee.List(xy).get(0),
                                            'y_pred': ee.List(xy).get(1)
                                        })
                                    )
                                # function to calculate log loss(need clarification)
                                def log_loss (pair_dict):
                                    pair_dict = ee.Dictionary(pair_dict)
                                    y = ee.Number(pair_dict.get('y_true'))
                                    p = ee.Number(pair_dict.get('y_pred'))
                                    #epsilon for numerical stability
                                    epsilon = 1e-15
                                    p_clip = p.max(epsilon).min(ee.Number(1).subtract(epsilon))
                                    # Log loss formula: -[y*log(p) + (1-y)*log(1-p)]
                                    loss = y.multiply(p_clip.log()).add(
                                        ee.Number(1).subtract(y).multiply(
                                            ee.Number(1).subtract(p_clip).log()
                                        )
                                    ).multiply(-1)
                                    return loss

                                #Calculate log losses for all test samples
                                loss_list = paired.map(log_loss)
                                avg_loss = ee.Number(loss_list.reduce(ee.Reducer.mean()))
                                return avg_loss
                            #mapped the log loss for all class
                            loss_list = class_list.map(per_class)
                            avg_loss_all = ee.Number(ee.List(loss_list).reduce(ee.Reducer.mean()))
                            #get actuall loss value:
                            act_loss = avg_loss_all.getInfo()

```


