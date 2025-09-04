import ee
ee.Initialize()
import pandas as pd
from tqdm import tqdm

class Hyperparameter_tuning:
    """
    Perform hyperparameter optimization for random forest classifiers. Several optimization are presented for different training data:
        1. Hard Classification tuning: This functions is used if the classification approach is hard multiclass classification
        2. Soft classification tuning: This functions is used if the classification approach is One-vs-Rest Binary classification framework
        3. Hard fold classification tuning: This function is used for multi-class classification with k-fold data
        4. Soft fold classification tuning: This functions is used if the classification approach is One-vs-Rest Binary classification framework with kfold data
    """
    def __init__(self):
        """
        Initialize the hyperparameter tuning class
        """
        pass
    ############################# 1. Multiclass Hard Classification Tuning ###########################
    def Hard_classification_tuning(self, train, test, image, class_property, 
                                   n_tree_list, var_split_list, min_leaf_pop_list):
        """
        Perform manual testing to find a set of parameters that yielded highest accuracy for Random Forest Classifier.
        Three main parameters were tested, namely Number of trees (n_tree), number of variable selected at split (var_split), and minimum sample population at leaf node (min_leaf_pop)
        This function is used for multiclass classification with training data from single or stratified random split
        Parameters:
            train: Training pixels
            test: Testing pixels
            image: ee.image for used for classification
            class_property: distinct labels in the training and testing data
            n_tree_list: list containing n_tree value for testing
            var_split_list: list containing var_split value for testing
            min_leaf_pop_list : list containing min_leaf_pop value for testing
        Returns:
        Best parameters combinations and resulting model accuracy (panda dataframe)
        """
        result = [] #initialize empty dictionary for storing parameters and accuracy score
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting hyperparameter tuning with {total_combinations} parameter combinations...")

        #manually test the classifiers, while looping through the parameters set
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
            #Convert the result into panda dataframe and print them
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
        
    ############################# 2. Binary One-vs-rest soft Classification Tuning ###########################
    def Soft_classification_tuning(self, train, test, image, class_property, 
                                   n_tree_list, var_split_list, min_leaf_pop_list, seed = 13):
        """
        Perform manual testing to find a set of parameters that yielded highest accuracy for Random Forest Classifier.
        Three main parameters were tested, namely Number of trees (n_tree), number of variable selected at split (var_split), and minimum sample population at leaf node (min_leaf_pop)
        This function is used for one-vs-rest binary classification with training data from single or stratified random split
        Parameters:
            train: Training pixels
            test: Testing pixels
            image: ee.image for used for classification
            class_property: distinct labels in the training and testing data
            n_tree_list: list containing n_tree value for testing
            var_split_list: list containing var_split value for testing
            min_leaf_pop_list : list containing min_leaf_pop value for testing
        Returns:
        Best parameters combinations and resulting model cross-entropy loss (panda dataframe)
        """
        #create an empty list to store all of the result
        result = []
        #get unique class ID
        class_list = train.aggregate_array(class_property).distinct()
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting soft classification tuning with {total_combinations} parameter combinations...")        
        #create a loop exploring all possible combination of parameter
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
                            #append the results of the tuning
                            result.append({
                                        'Number of Trees': n_tree,
                                        'Variable Per Split': var_split,
                                        'Minimum Leaf Populaton': min_leaf_pop,
                                        'Average Model Cross Entropy Loss': act_loss
                            })
                            print(f"Loss: {act_loss:.6f}")
                            pbar.update(1)

                            # Print this message if failed
                        except Exception as e:
                            print(f"Failed for Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                            print(f"Error: {e}")
                            pbar.update(1)
                            continue
            #convert the result into panda dataframe and viewed the best parameters
            if result:
                result_df = pd.DataFrame(result)
                result_df_sorted = result_df.sort_values(by='Average Model Cross Entropy Loss', ascending=True).reset_index(drop=True)
                
                print("\n" + "="*50)
                print("GRID SEARCH RESULTS")
                print("="*50)
                print("\nBest parameters (lowest loss):")
                print(result_df_sorted.iloc[0])
                print("\nTop 5 parameter combinations:")
                print(result_df_sorted.head())
                
                return result, result_df_sorted
            else:
                print("No successful parameter combinations found!")
                return [], pd.DataFrame
                #return [], pd.DataFrame()
    ############################# 3. Hard Multiclass Classification with k-fold data ###########################
    def hard_tuning_kfold(self, reference_fold, image, class_prop,  
                         n_tree_list, var_split_list, min_leaf_pop_list, tile_scale=16, pixel_size = 10):
        """
        Perform manual testing to find a set of parameters that yielded highest accuracy for Random Forest Classifier with stratified k-fold input data
        This function is used if stratified k-fold split is used to partitioned the samples
        parameters: 
            reference_fold: ee.featurecollection result from stratified kfold
            image: ee.image remote sensing data
            class_prop: property name for class labels
            n_tree_list: list of int, number of trees to test
            v_split_list: list of int, number of variables to test
            leaf_pop_list: list of int, minimum leaf population to test
            tile_scale: scale parameter for sampling
        return: list of dict with parameters and average accuracy
        """ 
        #define and set the previous fold result
        k = reference_fold.size().getInfo()
        fold_list = reference_fold.toList(k)
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting soft classification tuning with {total_combinations} parameter combinations...")        
        result = []
        # Pre-sample regions for each fold (optimization)
        print("Pre-sampling regions for all folds...")
        fold_samples = []
        for i in range(k):
            fold = ee.Feature(fold_list.get(i))
            training_fc = ee.FeatureCollection(fold.get('training'))
            testing_fc = ee.FeatureCollection(fold.get('testing'))
            
            train_pixels = image.sampleRegions(
                collection=training_fc,
                properties=[class_prop],
                scale=pixel_size,
                tileScale=tile_scale
            )
            test_pixels = image.sampleRegions(
                collection=testing_fc,
                properties=[class_prop],
                scale=pixel_size,
                tileScale=tile_scale
            )
            fold_samples.append({'train': train_pixels, 'test': test_pixels})
        print("Sampling complete!")        
        #Create a gridsearch tuning by manually looped through the parameter space
        with tqdm (total=total_combinations, desc="Hard K-Fold Tuning") as pbar:
            for n_tree in n_tree_list:
                for var_split in var_split_list:
                    for min_leaf_pop in min_leaf_pop_list:
                        print(f"Testing: Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                        fold_acc_list = []
                        for i in range (k):
                            try: 
                                
                                    #classifier set to multiclass hard classification
                                clf = ee.Classifier.smileRandomForest(
                                    numberOfTrees=n_tree,
                                    variablesPerSplit=var_split,
                                    minLeafPopulation=min_leaf_pop,
                                    seed=0
                                    ).train(
                                    features=train_pixels,
                                    classProperty=class_prop,
                                    inputProperties=image.bandNames()
                                    )
                                    #function to evaluate the model
                                classified_val = test_pixels.classify(clf)
                                model_val = classified_val.errorMatrix(class_prop, 'classification')
                                fold_accuracy = model_val.accuracy().getInfo()
                                fold_acc_list.append(fold_accuracy)
                                

                            except Exception as e:
                                print(f"Failed for fold {i}")
                                print(f"Failed for Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                                print(f"Error: {e}")
                                continue
                        
                        # Calculate average accuracy across all folds for this parameter combination
                        if fold_acc_list:
                            avg_acc = sum(fold_acc_list) / len(fold_acc_list)
                            #Put the result into a list
                            result.append({
                                'Number of Trees': n_tree,
                                'Variable Per Split': var_split,
                                'Minimum Leaf Population': min_leaf_pop,
                                'Average Model Accuracy': avg_acc
                            })
                            print(f"Average Accuracy: {avg_acc:.6f}")
                        else:
                            print(f"[WARNING] No valid accuracy scores for parameter combination Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                        pbar.update(1)                            
            #convert to panda data frame and print the result
        if result:
                result_pd = pd.DataFrame(result)
                result_pd_sorted = result_pd.sort_values(by='Average Model Accuracy', ascending=False).reset_index(drop=True)
                print("\n" + "="*50)
                print("GRID SEARCH RESULTS")
                print("="*50)
                print("\nBest parameters (highest accuracy):")
                print(result_pd_sorted.iloc[0])
                print("\nTop 5 Parameter combinations:")
                print(result_pd_sorted.head())         
                return result, result_pd_sorted
        else:
                print("No successful parameter combinations found!")
                return [], pd.DataFrame()
        
    def soft_tuning_kfold(self, folds, image, class_property, 
                          n_tree_list, var_split_list, min_leaf_pop_list, seed=0, pixel_size = 10, tile_scale=16):
        """
        Perform manual testing to find a set of parameters that yielded lowest cross-entropy loss for Random Forest Classifier with k-fold data.
        This function is used for one-vs-rest binary classification with k-fold cross-validation
        
        Parameters:
            folds: ee.FeatureCollection result from stratified k-fold split
            image: ee.Image remote sensing data for classification
            class_property: property name for class labels
            n_tree_list: list of int, number of trees to test
            var_split_list: list of int, number of variables per split to test
            min_leaf_pop_list: list of int, minimum leaf population to test
            seed: random seed for reproducibility
            tile_scale: scale parameter for sampling
            
        Returns:
            tuple: (result list, sorted DataFrame with parameters and average cross-entropy loss)
        """    
        #define and set the previous fold result
        k = folds.size().getInfo()
        fold_list = folds.toList(k)
        
        #get the list of unique class id
        first_fold = ee.Feature(fold_list.get(0))
        training_fc0 = ee.FeatureCollection(first_fold.get('training'))
        classes = training_fc0.aggregate_array(class_property).distinct().getInfo()
        print(f"Classes found: {classes}")
        
        result = []
        
        # Calculate total combinations for progress tracking
        total_combinations = len(n_tree_list) * len(var_split_list) * len(min_leaf_pop_list)
        print(f"Starting soft k-fold tuning with {total_combinations} parameter combinations across {k} folds...")
        
        # Pre-sample regions for each fold and each class (optimization)
        print("Pre-sampling regions for all folds and classes...")
        fold_class_samples = []
        for i in range(k):
            fold = ee.Feature(fold_list.get(i))
            training_fc = ee.FeatureCollection(fold.get('training'))
            testing_fc = ee.FeatureCollection(fold.get('testing'))
            
            class_samples = {}
            for class_id in classes:
                # Create binary training and testing data for this class
                binary_train = training_fc.map(lambda ft: ft.set(
                    'binary', ee.Algorithms.If(
                        ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                    )
                ))
                binary_test = testing_fc.map(lambda ft: ft.set(
                    'binary', ee.Algorithms.If(
                        ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                    )
                ))
                
                # Sample the image for this class
                train_pixels = image.sampleRegions(
                    collection=binary_train,
                    properties=['binary'],
                    scale=pixel_size,
                    tileScale=tile_scale
                )
                test_pixels = image.sampleRegions(
                    collection=binary_test,
                    properties=['binary'],
                    scale=pixel_size,
                    tileScale=tile_scale
                )
                
                class_samples[class_id] = {'train': train_pixels, 'test': test_pixels}
            
            fold_class_samples.append(class_samples)
        print("Sampling complete!")
        
        with tqdm(total=total_combinations, desc="Soft K-Fold Tuning") as pbar:
            for n_tree in n_tree_list:
                for var_split in var_split_list:
                    for min_leaf_pop in min_leaf_pop_list:
                     print(f"Testing: Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                    fold_loss_list = []

                    for i in range(k):
                        try:
                            class_loss_list = []
                            
                            for class_id in classes:
                                # Use pre-sampled data instead of sampling each time
                                train_pixels = fold_class_samples[i][class_id]['train']
                                test_pixels = fold_class_samples[i][class_id]['test']
                                
                                #Random Forest Model, set to probability mode
                                clf = (ee.Classifier.smileRandomForest(
                                        numberOfTrees=n_tree,
                                        variablesPerSplit=var_split,
                                        minLeafPopulation=min_leaf_pop,
                                        seed=seed)
                                        .setOutputMode('PROBABILITY'))
                                model = clf.train(
                                    features=train_pixels,
                                    classProperty='binary',
                                    inputProperties=image.bandNames()
                                )
                                test_classified = test_pixels.classify(model)
                               # Extract true class labels and predicted probabilities
                                y_true = test_classified.aggregate_array('binary')
                                y_pred = test_classified.aggregate_array('classification')
                                paired = y_true.zip(y_pred).map(
                                        lambda xy: ee.Dictionary({
                                            'y_true': ee.List(xy).get(0),
                                            'y_pred': ee.List(xy).get(1)
                                        })
                                    )
                                # function to calculate log loss
                                def log_loss(pair_dict):
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
                                loss_list = paired.map(log_loss)
                                avg_loss = ee.Number(loss_list.reduce(ee.Reducer.mean()))
                                class_loss = avg_loss.getInfo()
                                class_loss_list.append(class_loss)
                            
                            # Average loss across all classes for this fold
                            if class_loss_list:
                                fold_loss = sum(class_loss_list) / len(class_loss_list)
                                fold_loss_list.append(fold_loss)
                                print(f"Fold {i+1} Loss: {fold_loss:.6f}")

                        except Exception as e:
                            print(f"Failed for fold {i+1}")
                            print(f"Failed for Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                            print(f"Error: {e}")
                            continue
                    
                    # Calculate average loss across all folds for this parameter combination
                    if fold_loss_list:
                        avg_loss = sum(fold_loss_list) / len(fold_loss_list)
                        result.append({
                            'Number of Trees': n_tree,
                            'Variable Per Split': var_split,
                            'Minimum Leaf Population': min_leaf_pop,
                            'Average Model Cross Entropy Loss': avg_loss
                        })
                        print(f"Average Loss across folds: {avg_loss:.6f}")
                    else:
                        print(f"[WARNING] No valid loss scores for parameter combination Trees={n_tree}, Split={var_split}, Leaf={min_leaf_pop}")
                    
                    pbar.update(1)  # Update progress bar

        #convert the result into panda dataframe and view the best parameters
        if result:
            result_df = pd.DataFrame(result)
            result_df_sorted = result_df.sort_values(by='Average Model Cross Entropy Loss', ascending=True).reset_index(drop=True)
            
            print("\n" + "="*50)
            print("GRID SEARCH RESULTS")
            print("="*50)
            print("\nBest parameters (lowest loss):")
            print(result_df_sorted.iloc[0])
            print("\nTop 5 parameter combinations:")
            print(result_df_sorted.head())
            
            return result, result_df_sorted
        else:
            print("No successful parameter combinations found!")
            return [], pd.DataFrame()
