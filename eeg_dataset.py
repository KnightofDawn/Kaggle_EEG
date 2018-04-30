import numpy as np
import pandas as pd
import math
import os


class EEG_Dataset:

    def __init__(self, batch_size=32, batch_lag=128):
        self.features = []
        self.labels = []
        self.batch_size = batch_size
        self.batch_lag = batch_lag
        self.num_channels = 32
        self.batch_idx = 0

    def get_batch(self):
        #Need to lag each batch 
        batch_features = []
        batch_labels = []
        for time in range(min(self.batch_idx*self.batch_size,len(self.features)), 
            min(self.batch_idx*self.batch_size + self.batch_size, len(self.features))):
            lagged_feature = []
            for lag_idx in range(time-self.batch_lag+1,time+1):
                if lag_idx < 0:
                    # Pad with array of zeros (len = num_channels -> 32)
                    lagged_feature.append([0 for _ in range(0,self.num_channels)])
                else:
                    lagged_feature.append(self.features[lag_idx][:])
            batch_features.append(np.transpose(lagged_feature))
            batch_labels.append(self.labels[time][:])
        self.batch_idx += 1
        return np.expand_dims(np.array(batch_features),axis=3), np.array(batch_labels)


    def load_training_data(self, sub):
        n_subs = 12
        n_channels = 32
       
        data_dir = "Data/train"
        self.batch_idx = 0
        self.features = []
        self.labels = []
        print("Beginning to Load Data...")
        
        sub_data = []
        sub_label = []
        for series in range(1,7):
            print("Subject: ", sub, ", Series: ", series)
            csv = 'subj' + str(sub) + '_series' + str(series) + '_data.csv'
            with open(os.path.join(data_dir, csv)) as fstream:
                content = fstream.readlines()
            #print('---------------CONTENT for data------------------')
            #print(content[0])
            #print(content[1])
            content = [x.strip() for x in content]
            content = [x.split(',') for x in content]
            content = content[1:]
            content = [x[1:] for x in content]
            series_data = np.array(content)
            #print('----------------SERIES DATA -----------------------')
            #print(series_data[0])
            for line in series_data:
                self.features.append(line)
            
            csv = 'subj' + str(sub) + '_series' + str(series) + '_events.csv'
            with open(os.path.join(data_dir, csv)) as fstream:
                content = fstream.readlines()
            content = [x.strip() for x in content]
            content = [x.split(',') for x in content]
            content = content[1:]
            content = [x[1:] for x in content]
            #print('---------------CONTENT for labels------------------')
            #print(content[1][1:])
            series_labels = np.array(content)
            for line in series_labels:
                self.labels.append(line)

        #self.features = np.array(sub_data)
        #self.labels = np.array(sub_label)

        #print("-----FEATURES[0]------")
        #print(self.features[0])

        self.total_batches = math.ceil(len(self.labels)/self.batch_size)

        #np.save('eeg_train.npy', [data, label])

    def load_testing_data(self):
        n_subs = 12
        n_channels = 32
        n_series = [7,8]
        data_dir = "Data/train"
        self.batch_idx = 0

        self.features = []
        self.labels = []
        print("Beginning to Load Data...")
        
        
        for sub in range(1,13):
            sub_data = []
            sub_label = []
            for series in range(7,9):
                print("Subject: ", sub, ", Series: ", series)
                csv = 'subj' + str(sub) + '_series' + str(series) + '_data.csv'
                with open(os.path.join(data_dir, csv)) as fstream:
                    content = fstream.readlines()
                #print('---------------CONTENT for data------------------')
                #print(content[0])
                #print(content[1])
                content = [x.strip() for x in content]
                content = [x.split(',') for x in content]
                content = content[1:]
                content = [x[1:] for x in content]
                series_data = np.array(content)
                #print('----------------SERIES DATA-----------------------')
                #print(series_data[0])
                for line in series_data:
                    self.features.append(line)
                
                csv = 'subj' + str(sub) + '_series' + str(series) + '_events.csv'
                with open(os.path.join(data_dir, csv)) as fstream:
                    content = fstream.readlines()
                content = [x.strip() for x in content]
                content = [x.split(',') for x in content]
                content = content[1:]
                content = [x[1:] for x in content]
                #print('---------------CONTENT for labels------------------')
                #print(content[1][1:])
                series_labels = np.array(content)
                for line in series_labels:
                    self.labels.append(line)

                #self.features.append(sub_data)
                #self.labels.append(sub_label)


        return self.features, self.labels

        #np.save('eeg_train.npy', [data, label])


    def get_total_batches(self):
        return self.total_batches