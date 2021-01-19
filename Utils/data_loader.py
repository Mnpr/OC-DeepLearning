from keras.preprocessing.image import ImageDataGenerator

# Training Data Generator 
def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, img_dim=(256,256)):
    
    print("getting train generator...")
    
    # Keras @ImageDataGenerator Class
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # Keras @ImageDataGenerator(class).flow_form_dataframe(fn)
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=img_dim)
    
    return generator

# Test and Dev Data Generator
def get_test_and_dev_generator(dev_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, img_dim=(256,256)):

    print("getting train and valid generators...")
    
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=image_dir, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=img_dim)
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get dev generator
    dev_generator = image_generator.flow_from_dataframe(
            dataframe=dev_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=img_dim )
    
    # get test generator
    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=img_dim)
    
    return dev_generator, test_generator