import pandas as pd
import neattext.functions as nfx 

def data_csv():
    data = pd.read_csv("tweet_emotions.csv")
    return data

def clean_text(text):
        text = nfx.remove_stopwords(text)
        text = nfx.remove_punctuations(text)
        text = nfx.remove_special_characters(text)
        text = nfx.remove_dates(text)
        text = nfx.remove_emails(text)
        text = nfx.remove_emojis(text)
        text = nfx.remove_urls(text)
        text = nfx.remove_hashtags(text)
        text = nfx.remove_numbers(text)
        text = nfx.remove_userhandles(text)

        return(text)

def text_to_dataframe():
    data = []
    with open("data2.text") as f:
        lines = f.readlines()
        for line in lines:
            l =line.strip().split(";")
            data.append(l)
    return pd.DataFrame(data, columns = ["content", "sentiment"])

a = text_to_dataframe()
b = data_csv()
c= [a,b]
c = pd.concat(c)
b["clean-text"] = b["content"].apply(clean_text)
b.to_csv("clean_data.csv")
