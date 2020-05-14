import sys
import pathlib
import PIL
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
# from fool_image import fool_image

def main():

    # Load the model
    model = tf.keras.models.load_model('./models/incep3.h5')
    # model = VGG16(weights='imagenet', include_top=False)

    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])

    # Loop through the CSV file and make a prediction for each line
    with open('eval_classified_val.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for line in pathlib.Path(sys.argv[1]).open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(',')  # Extract CSV info

            print(image_id, image_path, image_height, image_width, image_channels)

            # Preprocess our data
            image_data = image.img_to_array(image.load_img(image_path, target_size=(128, 128)))
            preprocessed_image_data = image_data / 255.0

            # Generate a prediction
            prediction_probs = model.predict(np.expand_dims(preprocessed_image_data, 0))
            prediction = int(np.argmax(prediction_probs.reshape(-1)))

            # Write the prediction to the output file
            # print(prediction)
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[prediction]))


if __name__ == '__main__':
    main()
