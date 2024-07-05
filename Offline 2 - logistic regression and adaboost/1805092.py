import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np


######################################## LOGISTIC REGRESSION WEAK LEARNER ##############################

class LogisticRegressionWeakLearner:
    def __init__(self, learning_rate=0.01, n_iters=8000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, x, y):
        # Convert x to a NumPy array with float dtype
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)  # Ensure y is a NumPy array
        y = y.reshape(-1)
      
        n_samples, n_features = x.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(x, self.weights) + self.bias
       
            predictions = self.sigmoid_function(linear_model)
         #   print("predictions shape ", predictions.shape)

            # Calculate derivatives
            dw = (1 / n_samples) * np.dot(x.T, (predictions - y))
           # print("x shape ", x.shape, "predictions shape ", predictions.shape, "y shape ", y.shape)
            db = (1 / n_samples) * np.sum(predictions - y)

          
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        # Ensure x is a NumPy array with float dtype
        x = np.array(x, dtype=float)
        linear_model = np.dot(x, self.weights) + self.bias
        predictions = self.sigmoid_function(linear_model)
        predictions_cls = np.round(predictions).astype(int)

        return predictions_cls


 
 ############################### ADABOOST ##############################################################   

def is_dataframe(variable):
    return isinstance(variable, pd.DataFrame)

def AdaBoost(examples, Lweak, Knum):
    x= examples.x
    y= examples.y
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    #print(" x shape ", x.shape, "x type ", type(x)," y shape ", y.shape , "y type ", type(y))
    N = len(y)
    w = np.ones(N) / N  # Initialize example weights
   
    h = []
    z = []

    for k in range(Knum):
        # Resample examples with replacement based on weights
        indices = np.random.choice(N, size=N, replace=True, p=w)
        x_resampled, y_resampled = x[indices], y[indices]

        Lweak.__init__()
        # Train a weak learner using the resampled data
       
        weak_learner = Lweak
        weak_learner.fit(x_resampled, y_resampled)
        


        # Calculate the error of the hypothesis
        error = 0.00001

        for i in range(N):
            if weak_learner.predict(x[i]) != y[i]:
                error += w[i]
        if error > 0.5:
            #print("i ", i)
            continue
        for j in range(N):
            if weak_learner.predict(x[j]) == y[j]:
                w[j] *= error / (1 - error  )
        
        # Normalize the weights
        w /= np.sum(w)
        h.append(weak_learner.predict)
        z.append(np.log((1 - error) / (error )))
    
    return Weighted_Majority(h,z) 

def Weighted_Majority(h,z):
    def f(x):
        #print("h shape ", len(h), "z shape ", len(z))
        return np.sign(np.sum([z[i] * h[i](x) for i in range(len(h))]))
    return f


###################################### DATA PREPROCESSING HANDY FUNCTIONS ######################################################
def visualize_preprocessing(df, title):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Churn', data=df)
    plt.title(title)
    plt.show()

def feature_extraction_using_mutual_info(train_x, train_y,test_x, n_features):
    
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)
    train_y = pd.DataFrame(train_y)
    
    
    # Calculate mutual information scores
    if(train_x.isnull().values.any()):
        train_x = train_x.fillna(0)
    if(test_x.isnull().values.any()):
        test_x = test_x.fillna(0)
    if(train_y.isnull().values.any()):
        train_y = train_y.fillna(0)
        
    mutual_info_scores = mutual_info_classif(train_x, train_y)

    sorted_mutual_info_scores = pd.Series(mutual_info_scores, index=train_x.columns).sort_values(ascending=False)

    threshold= sorted_mutual_info_scores[n_features]
    columns_to_keep = sorted_mutual_info_scores[sorted_mutual_info_scores > threshold].index.tolist()
    
    train_x_selected = train_x[columns_to_keep]
    test_x_selected = test_x[columns_to_keep]
    '''
    for col in train_x_selected.columns:
        print("col ", col)
        print(train_x_selected.shape)
    '''
    return train_x_selected, test_x_selected

def labelEncode(df, label_features):
    labelEncoderX = LabelEncoder()
    for feature in label_features:
        df[feature] = labelEncoderX.fit_transform(df[feature])
    #print("label encoded df")
   # print(df.head())
    return df

def one_hot_encode(dataFrame, feature_names, one_hot_encoder = OneHotEncoder(sparse_output=False)):
    for feature in feature_names:
        encoded = one_hot_encoder.fit_transform(dataFrame[feature].values.reshape(-1, 1)).astype(np.int64)
        encoded_df = pd.DataFrame(encoded)
        encoded_df.columns = [feature + '_' + str(i) for i in range(encoded.shape[1])]
        encoded_df.index = dataFrame.index
        dataFrame = dataFrame.drop(feature, axis=1)
        dataFrame = pd.concat([dataFrame, encoded_df], axis=1)
    return dataFrame

def handleMissingValues(df, numerical_columns, categorical_columns):
    # Handling numerical missing values
    if numerical_columns:
        numerical_imputer = SimpleImputer(strategy='mean')
        df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

    if categorical_columns:
        # Handling categorical missing values
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    return df

def handle_extra_columns(train_x, test_x):
    # Identify extra columns in training data
    extra_columns_in_train = set(train_x.columns) - set(test_x.columns)

    # Add missing columns to test data and set values to 0
    for extra_column in extra_columns_in_train:
        test_x[extra_column] = 0

    return train_x, test_x

##############################################DATASET PREPROCESSING################################################
def preprocessTelco():
    # TRAIN DATA PREPROCESSING

    # Visualize data distribution before preprocessing
    # visualize_preprocessing(df, 'Before Preprocessing')
    df = pd.read_csv('telco\WA_Fn-UseC_-Telco-Customer-Churn copy.csv')

    # totalCharges is string so we need to convert it to numeric
    df['TotalCharges'] = df['TotalCharges'].str.replace(" ", "")

    # Convert to numeric, handle errors by setting to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['Churn'] = df['Churn'].fillna(df['Churn'].mode()[0])

    # Split data into features (x) and target (y)
    x = df.drop(columns=['Churn'])  # Features
    y = df['Churn']  # Target

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    
    
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    # Handling numerical missing values
    train_x = handleMissingValues(train_x, numerical_columns, categorical_columns)
    test_x = handleMissingValues(test_x, numerical_columns, categorical_columns)
    
    

    # Handling missing values in target column separately
    train_y = train_y.fillna(train_y.mode()[0])
    test_y = test_y.fillna(test_y.mode()[0])

    # Standardize (normalize) the numerical columns using mean and standard deviation
    scaler = StandardScaler()
    train_x[numerical_columns] = scaler.fit_transform(train_x[numerical_columns])
    test_x[numerical_columns] = scaler.transform(test_x[numerical_columns])


    label_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    one_hot_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    # label encode
    train_x = labelEncode(train_x, label_features)
    test_x = labelEncode(test_x, label_features)
    
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)
    test_y = label_encoder.fit_transform(test_y)
    

    # One-hot encode categorical columns
    
    train_x = one_hot_encode(train_x, one_hot_features)
    test_x = one_hot_encode(test_x, one_hot_features)

    # dropping customerID as it has unique values
    train_x.drop(['customerID'], axis=1, inplace=True)
    test_x.drop(['customerID'], axis=1, inplace=True)
    
    #save as a csv file
    train_x.to_csv('train_x.csv', index=False)



    return train_x, test_x, train_y, test_y

def preprocessAdult():
# Define header names based on your data
    header_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']

    # Read the CSV file with specified header names
    adult_train_df = pd.read_csv(r'adult\adult.data', header=None, names=header_names)
    adult_test_df = pd.read_csv(r'adult\adult.test', header=None, names=header_names, skiprows=1)  # skiprows=1 to skip the first row in the test file
    
   
   # Handle missing values in the target column


    #convert all >=50k to 1 and <50k to 0
    income_mapping_for_train = {'<=50K': 0, '>50K': 1}
    income_mapping_for_test = {'<=50K.': 0, '>50K.': 1}
   # print("Unique values in adult train:", adult_train_df['income'].unique())
    #print("Unique values in adult test:", adult_test_df['income'].unique())

    # Apply the mapping to the 'income' column after stripping spaces
    adult_train_df['income'] = adult_train_df['income'].str.strip().map(income_mapping_for_train)
    adult_test_df['income'] = adult_test_df['income'].str.strip().map(income_mapping_for_test)
    
    train_y = adult_train_df['income']
    test_y = adult_test_df['income']
    #print("train y ", train_y)
    #print("test y ", test_y)
    
    

    # Check if there are any missing values
    if train_y.dtype == 'O' and train_y[0] is np.nan:
        train_y = train_y.fillna(train_y.mode().iloc[0])
    
    if test_y.dtype == 'O' and test_y[0] is np.nan:
        test_y = test_y.fillna(test_y.mode().iloc[0])

    # Drop the target column from the features
 
    train_x = adult_train_df.drop(columns=['income']) # All columns except the last one
    test_x = adult_test_df.drop(columns=['income']) # All columns except the last one


    
   # print("adult train shape ", train_x.shape)
  #  print("adult test shape ", test_x.shape)

        
    numerical_columns = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    categorical_columns= ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    
    #handle missing values
    train_x = handleMissingValues(train_x, numerical_columns, categorical_columns)
    test_x = handleMissingValues(test_x, numerical_columns, categorical_columns)
    
    # Handling missing values in target column separately
    #train_y = train_y.fillna(train_y.mode().iloc[0])
   # test_y = test_y.fillna(test_y.mode().iloc[0])
    #label encode
    label_features=['sex']
    train_x = labelEncode(train_x, label_features)
    test_x = labelEncode(test_x, label_features)

    #one hot encode
    one_hot_features=['workclass', 'education' , 'marital-status' , 'occupation', 'relationship', 'race', 'native-country']
    train_x = one_hot_encode(train_x, one_hot_features)
    test_x = one_hot_encode(test_x, one_hot_features)
    
   # print("train x value before normalizing ", train_x.values.shape)
   # print("test x value before normalziing ", test_x.values.shape)
    
    train_x.to_csv('train_x.csv', index=False)
    test_x.to_csv('test_x.csv', index=False)

    #normalize
    scaler = StandardScaler()
    train_x[numerical_columns] = scaler.fit_transform(train_x[numerical_columns])
    test_x[numerical_columns] = scaler.transform(test_x[numerical_columns])
    
    train_x, test_x = handle_extra_columns(train_x, test_x)
    
    return train_x, test_x, train_y, test_y

def preprocessCreditCard():
    data_frame = pd.read_csv('creditcard\creditcard.csv')
    data_frame.drop(['Time'], axis = 1, inplace = True)
    #df = large_df.sample(frac=0.001, random_state=42)
    
    # Separate positive (fraudulent) and negative (non-fraudulent) samples
    positive_samples = data_frame[data_frame['Class'] == 1]
    negative_samples = data_frame[data_frame['Class'] == 0]
    
    
    
    # Choose a subset of 20,000 negative samples randomly
    negative_subset = negative_samples.sample(n=20000, random_state=42)

    # Combine positive samples with the randomly selected negative subset
    demo_subset = pd.concat([positive_samples, negative_subset])
    
    #handle missing values
    #demo_subset= handleMissingValues(demo_subset, demo_subset.columns, [])

    # Split the demo subset into features (X) and labels (y)
    x_demo = demo_subset.drop('Class', axis=1)  # Assuming 'Class' is the target variable
    y_demo = demo_subset['Class']

    
    # Adjust the test_size and random_state parameters as needed
    
    #visualize data distribution before preprocessing
    #visualize_preprocessing(df, 'Before Preprocessing')
 
    x_demo = x_demo.astype(float)
    y_demo = y_demo.astype(float)
    
    #one hot encode
    #x_demo = oneHotEncode(x_demo, [])

    #split the data
    train_x, test_x, train_y, test_y = train_test_split(x_demo,y_demo, test_size=0.2, random_state=42)

    numerical_columns = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                         'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                         'V21','V22','V23','V24','V25','V26','V27','V28','Amount']    
    #handle missing values
    train_x = handleMissingValues(train_x, numerical_columns, [])
    test_x = handleMissingValues(test_x, numerical_columns, [])
    

    train_x = one_hot_encode(train_x, [])
    test_x = one_hot_encode(test_x, []) 
    #normalize
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    
    #visualize data distribution after preprocessing
    #visualize_preprocessing(train_x, 'After Preprocessing train x')
    #visualize_preprocessing(test_x, 'After Preprocessing test x')
    
    

    train_x= pd.DataFrame(train_x)
    test_x= pd.DataFrame(test_x)
    #train_y= pd.DataFrame(train_y)
    #test_y= pd.DataFrame(test_y)
    
    return train_x, test_x, train_y, test_y

##################################### performance measures ################################################
def accuracy(y_true, y_pred):
   # y_true = y_true.reshape(y_true.shape[0], 1)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def sensitivity(y_true, y_pred): ##true positive rate / recall / hit rate
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    sensitivity = true_positives / (true_positives + false_negatives)
    return sensitivity

def specificity(y_true, y_pred): ##true negative rate
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity

def precision(y_true, y_pred): ## positive prediction value
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    precision = true_positives / (true_positives + false_positives)
    return precision

def false_discorvery_rate(y_true, y_pred):
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_discorvery_rate = false_positives / (false_positives + true_negatives)
    return false_discorvery_rate

def f1_score(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    sensitivity_value = sensitivity(y_true, y_pred)
    f1_score = 2 * (precision_value * sensitivity_value) / (precision_value + sensitivity_value)
    return f1_score




# Create examples object
class Examples:
    pass

#################################### MAIN FUNCTION ######################################################

if __name__ == '__main__':    
  
    train_x, test_x, train_y, test_y = preprocessTelco()
    train_x, test_x, train_y, test_y = preprocessAdult()
    train_x, test_x, train_y, test_y = preprocessCreditCard()
    
    train_x_selected, test_x_selected = feature_extraction_using_mutual_info(train_x, train_y, test_x ,n_features=10)
   
    
    ###################TO RUN LOGISTIC REGRESSION WEAK LEARNER ##############################

    print("------------------------------------------------------")
    print("Logistic Regression Weak Learner")
    classifier = LogisticRegressionWeakLearner()
    #scores = cross_val_score(classifier, train_x_selected, train_y, cv=5)
    #print("Cross-validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    examples = Examples()
    examples.x = train_x_selected.values 
    train_y_selected = train_y.astype(int)
    examples.y = train_y_selected
    
   # print("train x shape ", train_x_selected.shape)
   # print("train y shape ", train_y_selected.shape)

    classifier.fit(train_x_selected.values, train_y_selected)
    
    train_predictions = classifier.predict(train_x_selected.values)
    test_predictions= classifier.predict(test_x_selected.values)
    
    
    #Evaluate performance for train
    acc1 = accuracy(train_y_selected, train_predictions)
    sens1 = sensitivity(train_y_selected, train_predictions)
    spec1 = specificity(train_y_selected, train_predictions)
    prec1 = precision(train_y_selected, train_predictions)
    false1 = false_discorvery_rate(train_y_selected, train_predictions)
    f11 = f1_score(train_y_selected, train_predictions)
  
    print("train predictions")
    print(f"Accuracy: {acc1:.4f}")
    print(f"Sensitivity: {sens1:.4f}")
    print(f"Specificity: {spec1:.4f}")
    print(f"Precision: {prec1:.4f}")
    print(f"False Discorvery Rate: {false1:.4f}")
    print(f"F1 Score: {f11:.4f}")
    
    print("------------------------------------------------------")
    
    # Evaluate performance for test
    acc = accuracy(test_y, test_predictions)
    sens = sensitivity(test_y, test_predictions)
    spec = specificity(test_y, test_predictions)
    false = false_discorvery_rate(test_y, test_predictions)
    prec = precision(test_y, test_predictions)
    f1 = f1_score(test_y, test_predictions)


    print("test predictions")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"False Discorvery Rate: {false:.4f}")
    print(f"F1 Score: {f1:.4f}")


    
    #################### To run adaboost with different k values#####################
    print("------------------------------------------------------")
    print("AdaBoost with different k values 5,10,15,20")
    k_num_values =[5,10,15,20]
    
    for k_num in k_num_values:
        
        print("------------------------------------------------------")
        print("For Knum ", k_num)
        
        classifier = LogisticRegressionWeakLearner()
        #scores = cross_val_score(classifier, train_x_selected, train_y, cv=5)
        #print("Cross-validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        examples = Examples()
        examples.x = train_x_selected.values 
        train_y_selected = train_y.astype(int)
        examples.y = train_y_selected
        


        classifier.fit(train_x_selected.values, train_y_selected)
        
        
        ada = AdaBoost(examples, classifier, k_num)
  
        
        train_predictions = np.array([ada(x) for x in train_x_selected.values])
        test_predictions = np.array([ada(x) for x in test_x_selected.values])
        
        #Evaluate performance for train
        acc11 = accuracy(train_y_selected, train_predictions)
      
        print("train predictions")
        print(f"Accuracy: {acc11:.4f}")
        
        
        print("------------------------------------------------------")
        
        # Evaluate performance for test
        acc22 = accuracy(test_y, test_predictions)
        print("test predictions")
        print(f"Accuracy: {acc22:.4f}")
   
        
        
        
    

  

    


