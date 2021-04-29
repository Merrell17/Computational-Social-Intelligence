import pandas as pd
import math

df_train = pd.read_csv("training-part-2.csv")
df_test = pd.read_csv("test-part-2.csv")

# Remove white space present on facial muscles names
df_test.columns = df_test.columns.str.replace(' ', '')
df_train.columns = df_train.columns.str.replace(' ', '')

# Used to find the Standard Deviation of feature "i" in class "k" in gaussian
training_std = df_train.groupby('Class').std()

# Used to find the average of feature "i" in class "k" in gaussian
training_mean = df_train.groupby('Class').mean()

# List of each facial muscle ['AU_01 r', 'AU02 r' ... ]
facial_muscles = list(df_test.columns.values)[:-1]


# Function takes a facial expression 'smile' or 'frown' and a single row from the test set
# log p(x|Ck)
def gaussian_function(facial_expression, feature_vector):
    probability = 0
    # Loop over each Activation Unit
    for muscle in facial_muscles:
        # Find log 2pi * std deviation of class ik (left side)
        square_root2pi = math.sqrt((2 * math.pi))
        ls = math.log((square_root2pi * training_std[muscle][facial_expression]))

        # Find Euclidean Distance divided by 2 * standard deviation squared of class ik (right side)
        euclidean_distance = (feature_vector[muscle] - training_mean[muscle][facial_expression])**2
        std_dev_squared = 2 * (training_std[muscle][facial_expression]**2)
        rs = euclidean_distance / std_dev_squared

        # Take left side and right side and add it to our running total
        probability += rs + ls

    return -float(probability)

# Return Gaussian Decimal Error Rate
def error_rate():
    error_count = []
    # Pass each row[i] of the test data to the gaussian function
    for i in range((df_test.shape[0])):

        test_case = df_test.iloc[[i]]
        correct_class = test_case['Class'].tolist()[0]

        # Get posteriors of each class and choose one with greater posterior
        smile_posterior = gaussian_function('smile', test_case)
        frown_posterior = gaussian_function('frown', test_case)
        classification_decision = ('smile' if smile_posterior > frown_posterior else 'frown')

        # Check if classification matches the correct label and store result
        error_count.append(classification_decision == correct_class)

    return "Error Rate: {}%".format(error_count.count(False) / len(error_count) * 100)


print(error_rate())








