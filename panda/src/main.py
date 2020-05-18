from fedatabase import MongoDBConnect
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize

THRESHOLD = 0.0000000005
column_number = []

def filterColumns(column_names, selected_columns):
    i = 0

    temp = []
    for i in range(len(column_names)):
        if selected_columns[i] == True:
            temp.append(column_names[i])
            column_number.append(i)
        i += 1

    return temp

def calculate_scores(all_output, all_institutions):
    scores = []

    for i in range(len(all_output)):
        inst = {}
        inst['_id'] = all_institutions[i]['_id']
        x = all_output[i]
        sum = 0.0
        for value in x:
            sum += value

        inst['SCORE'] = sum
        scores.append(inst)

    return scores

def convert_to_feature_vector(samples):
    temp = []

    for x in samples:
        current_sample = []
        for k, v in x.items():
            if k != '_id' and k != 'SCORE':
                # Check if the value is a number, if not, set it to None (null)

                if type(v) == type(""):
                    try:
                        value = float(v)
                    except ValueError:
                        value = 0
                else:
                    value = 0

                current_sample.append(value)

        temp.append(current_sample[23:])

    return normalize(temp, norm="max")


def get_hbcus(all_institutions):

    temp = []

    for i in range(len(all_institutions)):
        x = all_institutions[i]
        if x["HBCU"] == "1":
            x["index"] = i
            temp.append(x)

    return temp

def get_hbcu_samples(hbcus, all_samples):
    temp = []

    for x in hbcus:
        temp.append(all_samples[x["index"]])

    return temp


if __name__ == "__main__":

    db = MongoDBConnect()

    # Get the all the institutions from the database
    all_institutions = db.get_all()
    all_samples = convert_to_feature_vector(all_institutions)
    # print("Institution Feature Vector Creation Successful")

    hbcus = get_hbcus(all_institutions)
    # print("Successfully gathered HBCUS")

    # Normalize values
    hbcu_samples = get_hbcu_samples(hbcus, all_samples)
    # print("Successfully gathered HBCU Feature Vectors")

    # Reduce the features
    selector = VarianceThreshold(THRESHOLD)
    selector.fit(hbcu_samples)
    # print("Successfully finished Variance Threshold")

    # Insert reduced feature vectors into output variable
    all_output = selector.transform(all_samples)
    # print("Successfully finished Transforming all institutions")

    # #Get column names and selected columns
    # column_names = db.get_variable_code()
    # selected_columns = list(selector.get_support())
    #
    # #Filter out unused columns
    # column_names = filterColumns(column_names, selected_columns)

    # print("Number of Institutions: " + str(len(all_samples)) + "\n")
    # print("Number of HBCUs: " + str(len(hbcu_samples)) + "\n")
    # print("Threshold: " + str(THRESHOLD) + "\n")
    # print("Original Number of Variables: " + str(len(hbcu_samples[0])) + "\n")
    # print("Selected Number of Variables: " + str(len(all_output[0])) + "\n")

    # scores = calculate_scores(all_output, all_institutions)
    # print("Scores Successfully Calculated for academicyear20182019")
    #
    # db.update_scores(scores)

    collection_name = "academicyear20172018"
    all_institutions = db.get_all_from_collection(collection_name)
    all_samples = convert_to_feature_vector(all_institutions)

    all_output = selector.transform(all_samples)
    #print("Successfully finished Transforming all institutions for " + collection_name)

    scores = calculate_scores(all_output, all_institutions)
    print("Scores Successfully Calculated for " + collection_name)

    db.update_scores_for_collection(collection_name, scores)

    collection_name = "academicyear20162017"
    all_institutions = db.get_all_from_collection(collection_name)
    all_samples = convert_to_feature_vector(all_institutions)

    all_output = selector.transform(all_samples)
    #print("Successfully finished Transforming all institutions for " + collection_name)

    scores = calculate_scores(all_output, all_institutions)
    print("Scores Successfully Calculated for " + collection_name)

    db.update_scores_for_collection(collection_name, scores)

    collection_name = "academicyear20152016"
    all_institutions = db.get_all_from_collection(collection_name)
    all_samples = convert_to_feature_vector(all_institutions)

    all_output = selector.transform(all_samples)
    #print("Successfully finished Transforming all institutions for " + collection_name)

    scores = calculate_scores(all_output, all_institutions)
    print("Scores Successfully Calculated for " + collection_name)

    db.update_scores_for_collection(collection_name, scores)