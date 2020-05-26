import pandas as pd

if __name__ == '__main__':

#Read DGA file
    dga_df = pd.read_csv("dga.csv", names=["DNS"])
    dga_df['Label']='DGA'
    #print(dga_df)
    x = dga_df.head(15000)
    print(x)

#Read benign file
    benign_df = pd.read_csv("benign.csv", names=["DNS"])
    benign_df['Label'] = 'Benign'
    #print(benign_df)
    y = benign_df.head(15000)
    print(y)

#Combine the datasets
    frames = [x, y]
    final_df = pd.concat(frames, ignore_index=True)
    print(final_df)
    final_df.to_csv('final_df.csv', index=False)
    final_df.sample(frac=1)
    final_df.to_csv(r'xgboost_data.csv', index = False)

