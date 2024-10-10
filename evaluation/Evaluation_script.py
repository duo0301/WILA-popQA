import pandas as pd
import json
import os
#from sklearn.metrics import precision_score, recall_score, f1_score

# the evaluation algorithme go like following ;
# 1- Read the results files : with following structure  
# lang(folder): model(folder): entity(file): {   idx: num; property:value;  answers: [ run0 answer, run1 answer, run2 answer], ground truth:value   } 
# 2- From the authors json file, Bulid pandas dataframe with the following structure 
# Input df = columns [promptlanguage, entitylanguage, modelname, entityID, Relid, Groundthruth (answer), pred ]
# Calculate the mertics : between the ground truth and the pred strings 
# output : df_results [ "propmtlanguage", entitylanguage, modelname, run_id, precision, recall, f1] 
            
def read_results_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
rootpath = "results-combined"

df_results = pd.DataFrame(columns=["Entity", "EntityLang", "PromptLang", "ModelName", "RunID", "Precision", "Recall", "F1"])

for lang_path in os.listdir(rootpath):    
    entity_language = lang_path.split("-")[0]
    for model_name in os.listdir(os.path.join(rootpath, lang_path)):
        for results_file_path in os.listdir(os.path.join(rootpath, lang_path, model_name)):    

            results_json = read_results_file(os.path.join(rootpath, lang_path, model_name, results_file_path))
            entity_id = results_file_path.split(".")[0].split("_")[0]
            prompt_language = results_file_path.split(".")[0].split("_")[-1]
            
            # line = [entity_id, entity_language, prompt_language, model_name]
            
            # for every question in the results file (13 questions)
            # and for every answer for the question let's do the evaluation for each run
            nb_runs = 3
            nb_questions = 13
            
            for num_question in range(1,nb_questions):
                for run_id in range(nb_runs):
                    # get the ground truth and the pred strings
                    ground_truth = results_json[str(num_question)]["ground_truth"]
                    pred = results_json[str(num_question)]["answers"][run_id]
                    # calculate the metrics
                    # precision : is the answer in the ground truth ? if yes it's 1 else 0
                    # recall : how many good answers appear in the ground truth


                    # questions with multiple answers : 6, 7, 10, 11, 13
                    # Tokenization of Strings
                    ground_truth_set = set(ground_truth.split(", "))
                    pred_set = set(pred.split(", "))

                    # Handling Empty Strings
                    # replace empty strings in the ground truth set and pred set with N/A
                    ground_truth_set = set(["N/A" if x == "" else x for x in ground_truth_set])
                    pred_set = set(["N/A" if x == "" else x for x in pred_set])

                    # Calculation of True Positives, False Positives, and False Negatives   
                    # Calculate precision, recall, and F1 score
                    true_positives = len(ground_truth_set & pred_set) #number of items that are in both the ground truth set and the prediction set.
                    false_positives = len(pred_set - ground_truth_set) #The number of items that are in the prediction set but not in the ground truth set.
                    false_negatives = len(ground_truth_set - pred_set) #The number of items that are in the ground truth set but not in the prediction set.

                    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                    
                    # Append results to the DataFrame
                    # Create a new row as a DataFrame
                    new_row = pd.DataFrame([{
                        "Entity": entity_id,
                        "EntityLang": entity_language,
                        "PromptLang": prompt_language,
                        "ModelName": model_name,
                        "RunID": run_id,
                        "QuestionID": num_question,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1
                    }])
                    
                    # Concatenate the new row to the results DataFrame
                    df_results = pd.concat([df_results, new_row], ignore_index=True)

                    # print(ground_truth_set, pred_set)
                    # print(df_results)
        df_avg_entity = df_results.groupby(['Entity', 'EntityLang', 'PromptLang', 'ModelName', 'RunID']).mean().reset_index()

        #  Exclude these non-numeric columns before calling the mean function. 
    columns_grp =  ["EntityLang", "PromptLang", "ModelName", "RunID"]
    columns_select = ["EntityLang", "PromptLang", "ModelName", "RunID", "Precision", "Recall", "F1"]
    df_avg_language = df_avg_entity[columns_select].groupby(columns_grp, as_index=False).mean()
    df_avg_language.to_csv("evaluation_results_by_language.csv", index=False)

# Average the results by entity
#df_avg_entity = df_results.groupby(['Entity', 'EntityLang', 'PromptLang', 'ModelName', 'RunID']).mean().reset_index()

print(df_avg_entity)
print(df_avg_language)


# Generate LaTeX table code for the final results (by language)
latex_table = df_avg_language.to_latex(index=False)
print(latex_table)
