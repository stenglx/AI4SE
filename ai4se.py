import pandas as pd 
import numpy as np 
import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score


def generate_code_embedding(model, code_example):
  nl_tokens=tokenizer.tokenize("return maximum value")
  print(code_example)
  code_tokens=tokenizer.tokenize(code_example, padding='max_length',truncation=True, max_length=508)#, padding='max_length', truncation=True, max_length=514)
  #  code_tokens=tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
  tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.eos_token] # 512 = total
  #tokens = tokenizer.encode(code_example, max_length=512, truncation=True)
  tokens_ids=tokenizer.convert_tokens_to_ids(tokens)

  print(tokens)
  print(tokens_ids)

  tokens_ids_tensor = torch.tensor(tokens_ids)
  print(tokens_ids_tensor)

  return model(tokens_ids_tensor[None,:])[0]

def getTrainingData():
    training_data = []
    training_labels = []
    count =0

    with open("data/primevul_train.jsonl", "r") as file:
        for line in file:
            data_line_json=line.strip()
            js=json.loads(data_line_json)
            #print(js)
            #break
            #data_train_df = pd.concat([data_train_df, pd.DataFrame(data_line_json)])
            #if not 'func' in js:
            #  print(js)
            #  break # TODO invesitage the issue here

            print(js['target'])
            embeddings = generate_code_embedding(model, js['func'])
            label = js['target']
            training_labels.append(label)
            training_data.append(embeddings[0, 0, :])
            count = count + 1 
            if count == 100:
                break

    return training_data, training_labels



    # generate dummy embeddings

    #embeddings = np.random.rand(5, 768)
    #labels = [1,0,1,0,0]

    test_labels = []
    test_embeddings = []
    with open("data/primevul_test.jsonl", "r") as file:
        for line in file:
            data_line_json=line.strip()
            js=json.loads(data_line_json)
            #print(js)
            #break
            #data_train_df = pd.concat([data_train_df, pd.DataFrame(data_line_json)])
            #if not 'func' in js:
            #  print(js)
            #  break # TODO invesitage the issue here
            embeddings = generate_code_embedding(model, js['func'])
            label = js['target']
            test_labels.append(label)
            test_embeddings.append(embeddings[0, 0, :].detach())         


    X = np.array(training_data)
    y = np.array(training_labels)

    rf = RandomForestClassifier()
    rf.fit(training_data, training_labels)


    y_pred = rf.predict(test_embeddings)


    print(classification_report(test_labels, y_pred))
    print("AUC: ",roc_auc_score(test_labels,y_pred))
    print("MCC: ",matthews_corrcoef(test_labels, y_pred))



if __name__ == '__main__':
    print("Main")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
   # tokenizer.model_max_length = 512
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    # model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    code = "Test code blub; x==1"
    embeddings = generate_code_embedding(model, code)

    #print(embeddings[0,0,:]) # get only the part with dimension 768

    labels = [1,1]

    training_data, training_labels = getTrainingData()
    #with torch.no_grad():
    #    rf = RandomForestClassifier()
    #    rf.fit(training_data, training_labels)#

     #   y_pred = rf.predict([embeddings[0,0,:], embeddings[0,0,:]])
     #   print(y_pred)


