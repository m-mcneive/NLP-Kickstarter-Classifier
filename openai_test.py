'''
KickStarter Project Group
Using the openAI API
'''import os
import openai
import json

# structure kickstarter data for OpenAI Api and split into Train, Dev, and Test sets
test_text = []
test_labels = []
dev_text=[]
dev_labels=[]

openai_results_dev = []
openai_results_test = []

def build_data():
     ## Load data
    kickstarterBlurb = [json.loads(line)['data']['blurb'] for line in open("data\kickstarter.json", "r", encoding="utf8")]
    kickstarterSuccess = [json.loads(line)['data']['state'] for line in open("data\kickstarter.json", "r", encoding="utf8")]
    size = len(kickstarterSuccess)
    for x in range( round(len(kickstarterSuccess) * 0.7) , round(len(kickstarterSuccess) * 0.7) + round(len(kickstarterSuccess) * 0.15) ):
        test_text.append(kickstarterBlurb[x])
        test_labels.append(kickstarterSuccess[x])
    for x in range( round(len(kickstarterSuccess) * 0.7) + round(len(kickstarterSuccess) * 0.15) , size):
        dev_text.append(kickstarterBlurb[x])
        dev_labels.append(kickstarterSuccess[x])
    #This structures the training date for OpenAI's API file,create function
    with jsonlines.open('data/train.jsonl', mode='w') as writer:
        for x in range(0, round(len(kickstarterSuccess) * 0.7)):
            writer.write({
                'text': kickstarterBlurb[x],
                'label': kickstarterSuccess[x]
                })
            
    with jsonlines.open('data/train.jsonl') as reader:
        for obj in reader:
            train_set.append(obj)
    return size

def init_openAI(): # Here we connect to openai api, need key to use properly
    openai.organization = "org-ZsbPtU8OmUNTsNKbKQ4kJi56"
    openai.api_key = os.environ['OPENAI_API_KEY']


if __name__ == "__main__":

    # Build data and check splits
    size = build_data()
    
    init_openAI()
    file = openai.File.create(file=open("data/train.jsonl", encoding = 'utf-8'), purpose="classifications")
    #print(file)
    #print(file['id'])
    
    for x in range(0, len(dev_labels)):
        result = openai.Classification.create(
            file= "file-pX9oOPHoUmiwNiPgsm7pazLg",
            query=dev_text[x],
            search_model="ada", 
            model="curie", 
            max_examples=2
        )
        openai_results_dev.append(result)


    # score the results of open ai vs actual 