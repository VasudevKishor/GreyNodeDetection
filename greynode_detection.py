#this does not contain the interface. This is just the code for the detection of malicious nodes

import pandas as pd  #pandas blah blah
from sklearn.model_selection import train_test_split  #train_test_split is used to split the data into training and testing sets [used in line 59]
from sklearn.preprocessing import StandardScaler  #StandardScaler is used to standardize the features by removing the mean and scaling to unit variance [used in line 61]
from sklearn.linear_model import LogisticRegression #LogisticRegression is used to predict the probability of a binary outcome [used in line 63]




"""I couldnt take the xlsx file as it was not present in the folder. So I have created a dummy data and used it for the
    code. I have also added the comments for the code. Please refer to the comments for the code explanation"""
try:
    data = pd.read_excel('/Users/vasudevkishor/Documents/StudyPersonal/Sharma Proj/node_data_with_trust_score.xlsx')
except UnicodeDecodeError as e:   #wont be necessary, just for safety !!!!!!!!!!!!!
    print("Error reading the Excel file:", e)
    print("Ensure the file is a valid Excel file and not corrupted.")
    exit()


# Select features and labels
features = ['traffic_ratio', 'normalized_loss', 'anomaly_score', 'sent_to_drop_ratio', 'node_trust_level']
"""
Relevance of the corresponding features:

- traffic_ratio: Ratio of incoming to outgoing traffic.
        --Malicious nodes often generate or attract abnormal traffic 
        (e.g., in DDoS attacks or data exfiltration).
          High or erratic traffic ratios can be a red flag.


- normalized_loss: Normalized loss rate.
        --High packet loss rates can indicate network congestion,
        hardware failures, or malicious attacks.
        Normalized loss rates help in comparing loss rates across different networks.

- anomaly_score: Anomaly detection score.
        --Anomaly detection algorithms assign scores to nodes based on their behavior.
        High anomaly scores indicate unusual or suspicious behavior.
        Malicious nodes often exhibit anomalous behavior.

- sent_to_drop_ratio: Ratio of sent packets to dropped packets.
        --Malicious nodes may send a large number of packets that are dropped by the network.
        High sent-to-drop ratios can indicate malicious intent or network misconfigurations.

- node_trust_level: Trust level of the node.
        --Trust levels can be assigned based on historical behavior, reputation, or authentication status.
        Low trust levels may indicate unverified or malicious nodes.
        High trust levels are assigned to nodes with a good reputation or verified identity.

"""
X = data[features] 
y = data['label_encoded']  
node_ids = data['node_id']
"""Please refer tp scaler stuff. Im not exactly sure what it does. I think it scales the data to a standard format and
    I am not sure why it does that. Nothing more"""
scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance
X_scaled = scaler.fit_transform(X) #same stuff. Do not have a proper idea about why this is here. Got this from Stack overflow (i guess)
 
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)


# Function to predict malicious nodes  
def detect_malicious_nodes(input_data, model, scaler): #Custom input can be given to predict malicious nodes [Parameter  : input_data]
    """
    Predicts malicious nodes based on custom input data.

    Args:
        input_data (DataFrame): Custom inputs containing features for prediction.
        model (LogisticRegression): Trained logistic regression model.
        scaler (StandardScaler): Scaler used for feature normalization.

    Returns:
        list: Node IDs predicted to be malicious.
    """
    # Normalize the custom input data
    input_features = input_data[features]
    input_scaled = scaler.transform(input_features)

    # Predict using the trained model
    predictions = model.predict(input_scaled)

    # Extract IDs of malicious nodes
    malicious_nodes = input_data.index[predictions == 1].tolist()  # Assuming index is the node ID
    return malicious_nodes
def cst_inp():
    tr = []
    nl = []
    ans = []
    stdr = []
    ntl = []
    index = []
    n = int(input("Enter the number of nodes: "))
    for i in range(n):
        print("Enter the details for Node", i+1)
        index.append(float(input("Enter the node ID: ")))
        tr.append(float(input("Enter the traffic ratio: ")))
        nl.append(float(input("Enter the normalized loss: ")))
        ans.append(float(input("Enter the anomaly score: ")))
        stdr.append(float(input("Enter the sent-to-drop ratio: ")))
        ntl.append(float(input("Enter the node trust level: ")))
    return tr, nl, ans, stdr, ntl, index

traffic_ratio, normalized_loss, anomaly_score, sent_to_drop_ratio, node_trust_level, index= cst_inp()

custom_input = pd.DataFrame({
    'traffic_ratio': traffic_ratio,
    'normalized_loss': normalized_loss,
    'anomaly_score': anomaly_score,
    'sent_to_drop_ratio': sent_to_drop_ratio,
    'node_trust_level': node_trust_level
}, index=index)

"""Here, the list index corresponds to the node ID. 
    eg : - Traffic_ratio[0] is for index[0] which is Node_1
         - Traffic_ratio[1] is for index[1] which is Node_2
         - Traffic_ratio[2] is for index[2] which is Node_3
         - Traffic_ratio[3] is for index[3] which is Node_4
         
         similarly for other features as well   !!!!!!!!!!!!"""
# The variable 'custom_input' is used to store user-defined input data. 
# This input can be customized based on the specific requirements of the program.
# For example, it could be a string, a list, a dictionary, or any other data type 
# that the program needs to process. The exact structure and purpose of 'custom_input' 
# would depend on the context in which it is used within the program.

malicious_nodes = detect_malicious_nodes(custom_input, model, scaler)  

"""Here the model is predefined as LogisticRegression
, and the scaler is predefined as StandardScaler."""

print("Malicious Nodes:", malicious_nodes)
