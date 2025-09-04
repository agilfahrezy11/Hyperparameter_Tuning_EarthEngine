import ee
ee.Initialize()

class FeatureExtraction:
    """
    Perform feature extraction as one of the input for land cover classification. Three types of split is presented here:
    Random Split: splitting the input data randomly based on specified split ratio
    stratified random split: splitting the input data randomly based on strata (lulc class). More representative for many cases
    statified k_fold split: splitting the input data into folds 
    """
    def __init__(self):
        """
        Initializing the class function
        """
############################# 1. Single Random Split ###########################
#extract pixel value for the labeled region of interest and partitioned them into training and testing data
#This can be used if the training/reference data is balanced across class and required more fast result
    def random_split(self, image, roi, class_property, split_ratio = 0.6, pixel_size = 10, tile_scale=16):
        """
        Perform single random split and extract pixel value from the imagery
            Parameters:
                image = ee.Image
                aoi = area of interest, ee.FeatureCollection
                split_ratio = 
            Returns:
                tuple: (training_samples, testing_samples)
        """
        #create a random column
        roi_random = roi.randomColumn()
        #partioned the original training data
        training = roi_random.filter(ee.Filter.lt('random', split_ratio))
        testing = roi_random.filter(ee.Filter.gte('random', split_ratio))
        #extract the pixel values
        training_pixels = image.sampleRegions(
                            collection=training,
                            properties = [class_property],
                            scale = pixel_size,
                            tileScale = tile_scale 
        )
        testing_pixels = image.sampleRegions(
                            collection=testing,
                            properties = [class_property],
                            scale = pixel_size,
                            tileScale = tile_scale 
        )
        print('Single Random Split Training Pixel Size:', training_pixels.size().getInfo())
        print('Single Random Split Testing Pixel Size:', testing_pixels.size().getInfo())
        return training_pixels, testing_pixels
    ############################## 2. Strafied Random Split ###########################
    # Conduct stratified train and test split, ideal for proportional split of the data
    def stratified_split (self, roi, image, class_prop, pixel_size= 10, train_ratio = 0.7, seed=0):
        """
        Used stratified random split to partitioned the original sample data into training and testing data used for model development
        Args:
            Split the region of interest using a stratified random approach, which use class label as basis for splitting
            roi: ee.FeatureCollection (original region of interest)
            class_prop: Class property (column) contain unique class ID
            tran_ratio: ratio for train-test split (usually 70% for training and 50% for testing)
        Return:
        ee.FeatureCollection, consist of training and testing data
        
        """
        #Define the unique class id using aggregate array
        classes = roi.aggregate_array(class_prop).distinct()
        #split the region of interest based on the class
        def split_class (c):
            subset = (roi.filter(ee.Filter.eq(class_prop, c))
                    .randomColumn('random', seed=seed))
            train = (subset.filter(ee.Filter.lt('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'training')))
            test = (subset.filter(ee.Filter.gte('random', train_ratio))
                        .map(lambda f: f.set('fraction', 'testing')))
            return train.merge(test)
        #map the function for all the class
        split_fc = ee.FeatureCollection(classes.map(split_class)).flatten()
        #filter for training and testing
        train_fc = split_fc.filter(ee.Filter.eq('fraction', 'training'))
        test_fc = split_fc.filter(ee.Filter.eq('fraction', 'testing'))
        print('Stratified Random Split Training Pixel Size:', train_fc.size().getInfo())
        print('Stratified Random Split Testing Pixel Size:', test_fc.size().getInfo())      
        #sample the image based stratified split data
        train_pix = image.sampleRegions(
                            collection=train_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16)
        test_pix = image.sampleRegions(
                            collection = test_fc,
                            properties = [class_prop],
                            scale = pixel_size,
                            tileScale = 16
        )
  
        return train_pix, test_pix
############################# 3. Stratified K-fold Split ###########################
# the strafied kfold cross validation split for more robust partitioning between training and validation data.
# Ideal for imbalance dataset. 
    def stratified_kfold(self, samples, class_property, k=5, seed=0):
        """
        Perform stratified kfold cross-validation split on input reference data.
        
        Parameters:
            samples (ee.FeatureCollection): training data or reference data which contain unique class label ID
            class_property (str): column name contain unique label ID
            k (int): Number of folds.
            seed (int): Random seed for reproducibility.
        
        Returns:
            ee.FeatureCollection: A collection of k folds. Each fold is a Feature
                                with 'training' and 'validation' FeatureCollections.
        """
        #define the logic for k-fold. It tells us how wide the split will be
        step = 1.0 / k
        #Threshold are similar to split ratio, in this context, an evenly space of data numbers. The results is a cut points for the folds,
        #in which each fold will takes sample whose asigned random number within the ranges
        thresholds = ee.List.sequence(0, 1 - step, step)
        #This code will aggregate unique class label into one distinct label
        classes = samples.aggregate_array(class_property).distinct()
        #function for create the folds using the given threshold
        def make_fold(threshold):
            threshold = ee.Number(threshold)
            #Split each class into training and testing, based on random numbers
            #each class split ensure startification during split
            def per_class(c):
                c = ee.Number(c)
                subset = samples.filter(ee.Filter.eq(class_property, c)) \
                                .randomColumn('random', seed)
                training = subset.filter(
                    ee.Filter.Or(
                        ee.Filter.lt('random', threshold),
                        ee.Filter.gte('random', threshold.add(step))
                    )
                )
                testing = subset.filter(
                    ee.Filter.And(
                        ee.Filter.gte('random', threshold),
                        ee.Filter.lt('random', threshold.add(step))
                    )
                )
                return ee.Feature(None, {
                    'training': training,
                    'testing': testing
                })
            #Applied the splits for each class in the dataset
            splits = classes.map(per_class)
            # merge all classes back together for this fold
            # merged all classes in the training subset
            training = ee.FeatureCollection(splits.map(lambda f: ee.Feature(f).get('training'))).flatten()
            # merge all classes in the testing subset
            testing = ee.FeatureCollection(splits.map(lambda f: ee.Feature(f).get('testing'))).flatten()
            return ee.Feature(None, {'training': training, 'testing': testing})

        folds = thresholds.map(make_fold)
        # Print overall k-fold information (moved outside the mapped function)
        print(f'Created {k} folds for stratified k-fold cross-validation')
        print('Total input samples:', samples.size().getInfo())
        print(f'Each fold will have approximately {samples.size().divide(k).getInfo():.0f} samples for validation')        
        return ee.FeatureCollection(folds)
    ############################# 4. Inspecting the folds ###########################
    # Used to inspect the resulr of stratified k-fold size
    def inspect_fold(self, folds, fold_index):
        """Inspect a specific fold's sizes"""
        fold = ee.Feature(folds.toList(1, fold_index).get(0))
        train = ee.FeatureCollection(fold.get('training'))
        val = ee.FeatureCollection(fold.get('testing'))
        print(f'Fold {fold_index + 1} - Training: {train.size().getInfo()}, Testing: {val.size().getInfo()}')
        return train, val  