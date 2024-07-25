import numpy as np
import pandas as pd
import tensorflow as tf

unique_chars = pd.read_csv("models/keras/example_model_vocab.csv").char.to_list()

###     vvv     Code from handout notebook      vvv     ###

# Character to numeric value dictionary
char_to_num = tf.keras.layers.StringLookup(vocabulary=unique_chars, mask_token=None)

# Reverse dictionary
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

n_classes = len(unique_chars)

MAX_LABEL_LENGTH = 17
IMG_WIDTH = 200
IMG_HEIGHT = 50
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16


def load_image(file_name: str):
    """
    This function loads and preprocesses images. It first receives the image path, which is used to
    decode the image as a JPEG using TensorFlow. Then, it converts the image to a tensor and applies
    two processing functions: resizing and normalization. The processed image is then returned by
    the function.

    Argument :
        file_name : The path of the image file to be loaded.

    Return:
        image : The loaded image as a tensor.
    """

    # Read the Image
    image = tf.io.read_file(file_name)

    # Decode the image
    decoded_image = tf.image.decode_jpeg(contents=image, channels=1)

    # Convert image data type.
    cnvt_image = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)

    # Resize the image
    resized_image = tf.image.resize(images=cnvt_image, size=(IMG_HEIGHT, IMG_WIDTH))

    # Transpose
    image = tf.transpose(resized_image, perm=[1, 0, 2])

    # Convert image to a tensor.
    image = tf.cast(image, dtype=tf.float32)

    # Return loaded image
    return image


def decode_pred(pred_label):
    """
    The decode_pred function is used to decode the predicted labels generated by the OCR model.
    It takes a matrix of predicted labels as input, where each time step represents the probability
    for each character. The function uses CTC decoding to decode the numeric labels back into their
    character values. The function also removes any unknown tokens and returns the decoded texts as a
    list of strings. The function utilizes the num_to_char function to map numeric values back to their
    corresponding characters. Overall, the function is an essential step in the OCR process, as it allows
    us to obtain the final text output from the model's predictions.

    Argument :
        pred_label : These are the model predictions which are needed to be decoded.

    Return:
        filtered_text : This is the list of all the decoded and processed predictions.

    """

    # Input length
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]

    # CTC decode
    decode = tf.keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0][
        :, :MAX_LABEL_LENGTH
    ]

    # Converting numerics back to their character values
    chars = num_to_char(decode)

    # Join all the characters
    texts = [tf.strings.reduce_join(inputs=char).numpy().decode("UTF-8") for char in chars]

    # Remove the unknown token
    filtered_texts = [text.replace("[UNK]", " ").strip() for text in texts]

    return filtered_texts


def encode_single_sample(file_name: str, label: str = ""):
    """
    The function takes an image path and label as input and returns a dictionary containing the processed image tensor and the label tensor.
    First, it loads the image using the load_image function, which decodes and resizes the image to a specific size. Then it converts the given
    label string into a sequence of Unicode characters using the unicode_split function. Next, it uses the char_to_num layer to convert each
    character in the label to a numerical representation. It pads the numerical representation with a special class (n_classes)
    to ensure that all labels have the same length (MAX_LABEL_LENGTH). Finally, it returns a dictionary containing the processed image tensor
    and the label tensor.

    Arguments :
        file_name : The location of the image file.
        label      : The text to present in the image.

    Returns:
        dict : A dictionary containing the processed image and label.
    """

    # Get the image
    image = load_image(file_name)

    # Convert the label into characters
    chars = tf.strings.unicode_split(label, input_encoding="UTF-8")

    # Convert the characters into vectors
    vecs = char_to_num(chars)

    # Pad label
    pad_size = MAX_LABEL_LENGTH - tf.shape(vecs)[0]
    vecs = tf.pad(vecs, paddings=[[0, pad_size]], constant_values=n_classes + 1)

    return {"image": image, "label": vecs}


###     ^^^     Code from handout notebook      ^^^     ###


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    model = tf.keras.models.load_model("models/keras/example_model.keras")
    ds = (
        tf.data.Dataset.from_tensor_slices(
            (np.array(df.file_name.to_list()), np.array([""] * len(df)))
        )
        .map(encode_single_sample, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    preds = model.predict(ds)
    preds = decode_pred(preds)
    df = df.copy()
    df["prediction"] = preds
    return df
