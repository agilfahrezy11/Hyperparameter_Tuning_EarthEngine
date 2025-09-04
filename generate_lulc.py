import ee
import pandas as pd
ee.Initialize()

class Generate_and_evaluate_LULC:
    def __init__(self):
        """
        Perform classification to generate Land Cover Land Use Map. The parameters used in the classification should be the result of hyperparameter tuning
        """
        ee.Initialize()
        pass

    ############################# 1. Multiclass Classification ###########################
    def multiclass_classification(self, training_data, class_property, image, ntrees = 100, 
                                  v_split = None, min_leaf = 1, seed=0):
        """
        Perform multiclass hard classification to generate land cover land use map
            Parameters:
            training data: ee.FeatureCollection, input sample data from feature extraction function (must contain pixel value)
            class_property (str): Column name contain land cover class id
            ntrees (int): Number of trees (user should input the best parammeter from parameter optimization)
            v_split (int): Variables per split (default = sqrt(#covariates)). (user should input the best parammeter from parameter optimization)
            min_leaf (int): Minimum leaf population. (user should input the best parammeter from parameter optimization)
            seed (int): Random seed.
        returns:
        ee.Image contain hard multiclass classification
        """
   # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        #if for some reason var split is not specified, used 
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()
        #Random Forest model
        clf = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees, 
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed)
        model = clf.train(
            features=training_data,
            classProperty=class_property,
            inputProperties=image.bandNames()
        )
        #Implement the trained model to classify the whole imagery
        multiclass = image.classify(model)
        return multiclass
     ############################# 1. One-vs-rest (OVR) binary Classification ###########################
    def ovr_classification(self, training_data, class_property, image, include_final_map=True,
                                ntrees = 100, v_split = None, min_leaf =1, seed=0, probability_scale = 100):
        """
        Implementation of one-vs-rest binary classification approach for multi-class land cover classification, similar to the work of
        Saah et al 2020. This function create probability layer stack for each land cover class. The final land cover map is created using
        maximum probability, via Argmax

        Parameters
            training_data (ee.FeatureCollection): The data which already have a pixel value from input covariates
            class_property (str): Column name contain land cover class id
            image (ee.Image): Image data
            covariates (list): covariates names
            ntrees (int): Number of trees (user should input the best parammeter from parameter optimization)
            v_split (int): Variables per split (default = sqrt(#covariates)). (user should input the best parammeter from parameter optimization)
            min_leaf (int): Minimum leaf population. (user should input the best parammeter from parameter optimization)
            seed (int): Random seed.
            probability scale = used to scaled up the probability layer

        Returns:
            ee.Image: Stacked probability bands + final classified map.
        """
        # parameters and input valdiation
        if not isinstance(training_data, ee.FeatureCollection):
            raise ValueError("training_data must be an ee.FeatureCollection")
        if not isinstance(image, ee.Image):
            raise ValueError("image must be an ee.Image")
        #if for some reason var split is not specified, used 
        if v_split is None:
            v_split = ee.Number(image.bandNames().size()).sqrt().ceil()
        
        # Get distinct classes ID from the training data. It should be noted that unique ID should be in integer, since 
        # float types tend to resulted in error during the band naming process 
        class_list = training_data.aggregate_array(class_property).distinct()
        
        #Define how to train one vs rest classification and map them all across the class
        def per_class(class_id):
            class_id = ee.Number(class_id)
            #Creating a binary features, 1 for a certain class and 0 for other (forest = 1, other = 0)
            binary_train = training_data.map(lambda ft: ft.set('binary', ee.Algorithms.If(
                            ee.Number(ft.get(class_property)).eq(class_id), 1, 0
                                )
                            ))
            #Build random forest classifiers, setting the outputmode to 'probability'. The probability mode will resulted in
            #one binary classification for each class. This give flexibility in modifying the final weight for the final land cover
            #multiprobability resulted in less flexibility in modifying the class weight
            #(the parameters required tuning)
            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=ntrees, 
                variablesPerSplit=v_split,
                minLeafPopulation=min_leaf,
                seed=seed
            ).setOutputMode("PROBABILITY")
            #Train the model
            trained = classifier.train(
                features=binary_train,
                classProperty="binary",
                inputProperties=image.bandNames()
            )
            # Apply to the image and get the probability layer
            # (probability 1 represent the confidence of a pixel belonging to target class)
            prob_img = image.classify(trained).select(['probability_1'])
            #scaled and convert to byte
            prob_img = prob_img.multiply(probability_scale).round().byte()
            #rename the bands
            #Ensure class_id is integer. 
            class_id_str = class_id.int().format()
            band_name = ee.String ('prob_').cat(class_id_str)

            return prob_img.rename(band_name)
        # Map over classes to get probability bands
        prob_imgs = class_list.map(per_class)
        prob_imgcol = ee.ImageCollection(prob_imgs)
        prob_stack = prob_imgcol.toBands()

        #if final map  is not needed, the functin will return prob bands only
        if not include_final_map:
            return prob_stack
        #final map creation using argmax
        print('Creating final classification using argmax')
        class_ids = ee.List(class_list)
        #find the mad prob in each band for each pixel
        #use an index image (0-based) indicating which class has highest probability
        max_prob_index = prob_stack.toArray().arrayArgmax().arrayGet(0)

        #map the index to actual ID
        final_lc = max_prob_index.remap(ee.List.sequence(0, class_ids.size().subtract(1)),
                                        class_ids).rename('classification')
        #calculate confidence layer
        max_confidence = prob_stack.toArray().arrayReduce(ee.Reducer.max(), [0]).arrayGet([0]).rename('confidence')
        #stack the final map and confidence
        stacked = prob_stack.addBands([final_lc, max_confidence])
        return stacked

    def thematic_assessment(self, lcmap, validation_data, class_property,
                            region=None, scale=10, return_per_class = True):
        """
        Evaluate the thematic accuracy of land cover land use map wih several accuracy metric:
        overall accuracy
        F1-score
        Geometric mean
        Per-class accuracy
        balanced accuracy score
        
        """
        if region is None:
            validation_sample = lcmap.select('classification').sampleRegions(
                collection = validation_data,
                properties = [class_property],
                scale = scale,
                geometries = False,
                tileScale = 4
            )
        else:
            validation_sample = lcmap.select('classification').sampleRegions(
                collection = validation_data.filterBounds(region),
                properties = [class_property],
                scale = scale,
                tileScale = 4
            )
        #create a confuction matrix
        confusion_matrix = validation_sample.errorMatrix(class_property, 'classification')
        #basic metric calculation:
        oa = confusion_matrix.accuracy()
        kappa = confusion_matrix.kappa()
        #calculate per class accuracy
        class_order = confusion_matrix.order()
        matrix_array = confusion_matrix.array()
        def per_class_acc():
            """
            calculate precision, recal, and f1 per class
            """
            n_class = matrix_array.length().get(0)
            def per_class_calc(i):
                i = ee.Number(i)
                #TP = True Positive
                tp = matrix_array.get([i, i])
                #FP = False positive
                col_sum = matrix_array.slice(0,0,-1).slice(1, i, i.add(1)).reduce(ee.Reuducer.sum(), [0])
                fp = col_sum.get([0,0]).subtract(tp)
                #fn = false negatives
                row_sum = matrix_array.slice(0, i, i.add(1)).slice(1,0,-1).reduce(ee.Reducer.sum(), [1])
                fn = row_sum.get([0,0]).subtract(tp)
                #calculate precision (user accuracy), recall (producer accuracy), and f1-score
                precision = ee.Number(tp).divide(ee.Number(tp).add(ee.Number(fp)))
                recall = ee.Number(tp).divide(ee.Number(tp).add(ee.Number(fn)))
                #f1 score with zero division handling
                f1 = ee.Algorithms.If(precision.add(recall).eq(0), 0,
                                    precision.multiply(recall).multiply(2).divide(precision.add(recall))
                                    )
                return ee.Dictionary({
                    'class_ID': class_order.get(i),
                    'Preicison/User Accuracy': precision, 
                    'Recall/Producer Accuracy': recall,
                    'F1 Score': f1,
                    'True Positive': tp,
                    'False Positive': fp,
                    'False Negative': fn
                })
            #map all classes
            class_index = ee.List.sequence(0, n_class.subtract(1))
            per_class_result = class_index.map(per_class_calc)
            return per_class_result
        
        #now calculate the metric:
        per_class_metrics = per_class_acc()
            # Calculate macro-averaged metrics
        def calculate_macro_metrics():
            """Calculate macro-averaged F1, Precision, Recall"""
            
            # Extract individual metric lists
            precision_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('precision'))
            recall_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('recall'))
            f1_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('f1_score'))
            
            # Calculate means
            macro_precision = ee.Array(precision_values).reduce(ee.Reducer.mean(), [0]).get([0])
            macro_recall = ee.Array(recall_values).reduce(ee.Reducer.mean(), [0]).get([0])
            macro_f1 = ee.Array(f1_values).reduce(ee.Reducer.mean(), [0]).get([0])
            
            return {
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1
            }
        
        # Calculate micro-averaged metrics
        def calculate_micro_metrics():
            """Calculate micro-averaged F1, Precision, Recall"""
            
            # Sum all TP, FP, FN across classes
            total_tp = ee.Array(per_class_metrics.map(lambda x: ee.Dictionary(x).get('tp'))).reduce(ee.Reducer.sum(), [0]).get([0])
            total_fp = ee.Array(per_class_metrics.map(lambda x: ee.Dictionary(x).get('fp'))).reduce(ee.Reducer.sum(), [0]).get([0])
            total_fn = ee.Array(per_class_metrics.map(lambda x: ee.Dictionary(x).get('fn'))).reduce(ee.Reducer.sum(), [0]).get([0])
            
            # Calculate micro metrics
            micro_precision = ee.Number(total_tp).divide(ee.Number(total_tp).add(ee.Number(total_fp)))
            micro_recall = ee.Number(total_tp).divide(ee.Number(total_tp).add(ee.Number(total_fn)))
            micro_f1 = micro_precision.multiply(micro_recall).multiply(2).divide(micro_precision.add(micro_recall))
            
            return {
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1
            }
        
        # Calculate Geometric Mean
        def calculate_geometric_mean():
            """Calculate Geometric Mean of per-class recalls (sensitivities)"""
            
            recall_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('recall'))
            
            # Calculate geometric mean: (r1 * r2 * ... * rn)^(1/n)
            # Using log transform: exp(mean(log(recalls)))
            log_recalls = ee.Array(recall_values).log()
            mean_log_recall = log_recalls.reduce(ee.Reducer.mean(), [0]).get([0])
            geometric_mean = ee.Number(mean_log_recall).exp()
            
            return geometric_mean
        
        # Calculate Balanced Accuracy
        def calculate_balanced_accuracy():
            """Calculate Balanced Accuracy (macro-averaged recall)"""
            recall_values = per_class_metrics.map(lambda x: ee.Dictionary(x).get('recall'))
            balanced_accuracy = ee.Array(recall_values).reduce(ee.Reducer.mean(), [0]).get([0])
            return balanced_accuracy
        
        # Get all calculated metrics
        macro_metrics = calculate_macro_metrics()
        micro_metrics = calculate_micro_metrics()
        geometric_mean = calculate_geometric_mean()
        balanced_accuracy = calculate_balanced_accuracy()
        
        results = {
            'confusion_matrix': confusion_matrix,
            'class_order': class_order,
            'overall accuracy': oa,
            'balanced accuracy': balanced_accuracy,
            'kappa': kappa,
            'macro_f1': macro_metrics['macro_f1'],
            'macro_precision': macro_metrics['macro_precision'], 
            'macro_recall': macro_metrics['macro_recall'],
            'micro_f1': micro_metrics['micro_f1'],
            'micro_precision': micro_metrics['micro_precision'],
            'micro_recall': micro_metrics['micro_recall'],
            'geometric_mean': geometric_mean,
        }
        if return_per_class: 
            results['per_class_metric'] = per_class_metrics
        return results
    ############################# 8. Printing the accuracy metrics ###########################

    def print_metrics (self, evaluation, class_names = None):
        
        print("CLASSIFICATION THEMATIC EVALUATION RESULT\n")
        #overall metrics
        print('Overall Metrics:')
        print(f'Overall Accuracy:{evaluation["overall_accuracy"].getInfo():.4f}')
        print(f'Balanced Accuracy: {evaluation["balanced_accuracy"].getInfo():.4f}')
        print(f'Kappa Coefficient:{evaluation["kappa"].getInfo():.4f}')
        print(f'Geometric Mean: {evaluation['geometric_mean'].getInfo():.4f}')

        #Aggregate Metric
        print('Aggregate Metric:')
        print(f"Macro F-1 Score: {evaluation['macro_f1'].getInfo():.4f}")
        print(f"Micro F1-Score: {evaluation['micro_f1'].getInfo():.4f}")
        print(f"Macro Precision: {evaluation['macro_precision'].getInfo():.4f}")
        print(f"Macro Recall: {evaluation['macro_recall'].getInfo():.4f}")

        #per class if requested
        if 'per_class_metrics' in evaluation:
            print('PER-CLASS METRICS:')
            per_class = evaluation['per_class_metrics'].getInfo()
            print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print('-' * 45)
            for metrics in per_class:
                class_id = metrics['class_id']
                class_names = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
                print(f"{class_names:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        print("\n=== METRIC INTERPRETATIONS ===")
        print("• Overall Accuracy: Can be misleading with imbalanced data")
        print("• Balanced Accuracy: Average of per-class recalls (better for imbalanced data)")
        print("• Macro F1: Unweighted average F1 across classes (treats all classes equally)")
        print("• Micro F1: Weighted by class frequency (dominated by frequent classes)")
        print("• Geometric Mean: Sensitive to poor performance on any class")
        print("• Kappa: Agreement beyond chance, accounts for class imbalance")    
    