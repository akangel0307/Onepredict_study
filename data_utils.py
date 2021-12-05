import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_dataframe(PATH, stage, stage_x):
    # get x and y from corresponding dirs
    path = os.path.join(PATH, stage_x)                                     #경로 이름이 연결되고 파일 경로가 획득됩니다. PATH\stage_x
    dataframe_x = pd.DataFrame()
    for file_name in os.listdir(path):
        dataframe = pd.read_csv(os.path.join(path, file_name))
        dataframe = dataframe.drop(columns=['TIMESTAMP','MACHINE_ID','MACHINE_DATA'])                  #'TIMESTAMP'데이터는 의미 없는 평균 데이터, 삭제
        dataframe_x = dataframe_x.append(dataframe,ignore_index=True)
    # dataframe_group_x = dataframe_x.groupby(['WAFER_ID','STAGE'])
    y_path = os.path.join(PATH, "CMP-"+stage+"-removalrate.csv")           #CMP-test-removalrate.csv
    dataframe_y = pd.read_csv(y_path)

    dataframe_y = dataframe_y.loc[dataframe_y['AVG_REMOVAL_RATE'] <= 1000] #1000보다 큰 데이터는 이상값으로 간주되어 폐기될 수 있습니다.
    dataframe_y.hist('AVG_REMOVAL_RATE')                                   #이 열에 대한 분포 히스토그램을 그립니다.
    # plt.hist(dataframe_y['AVG_REMOVAL_RATE'])                            #이 데이터 열을 플로팅합니다. 이 데이터 열은 두 섹션으로 명확하게 나뉩니다.
    # plt.show()

    print("dataframe_x.shape",dataframe_x.shape)
    print("dataframe_y.shape", dataframe_y.shape)
    return dataframe_x, dataframe_y
    # return

# dataframe_x, dataframe_y = load_dataframe(PATH, stage, stage_x)

def abstract_statistics(dataframe_x, dataframe_y, statistics=['mean','std','min','median','max']):
    # abstract statistics for virtual metrology
    # dataframe_x has dropped timestamps
    dataframe_group_x = dataframe_x.groupby(['WAFER_ID','STAGE'])                       # 그룹 데이터

    dataframe_statistics = dataframe_group_x.agg(statistics)                            # 그룹핑 후 각 차원 데이터의 관련 통계 계산
    # print("dataframe_statistics",dataframe_statistics)
    # dataframe_statistics.to_csv("dataframe_statistics.csv", index=False, sep=',')     # 위의 관련 통계를 csv 파일에 씁니다.

    columns = dataframe_x.columns                                                       # 원본 데이터의 열 이름
    dataframe_statistics.columns = generate_columns_name(columns, statistics)           # 위의 관련 통계 데이터에 대한 새 열 이름 생성
    dataframe_statistics = pd.DataFrame(dataframe_statistics)
    dataframe_statistics.reset_index(inplace=True)                                      # 인덱스 복원 및 누락된 데이터 채우기
    # dataframe_statistics.to_csv("dataframe_statistics_final.csv", index=False, sep=',') # 위의 관련 통계를 csv 파일에 씁니다.

    # data = pd.concat([dataframe_statistics, dataframe_y], ignore_index=True)
    data = pd.merge(dataframe_statistics, dataframe_y)
    # data.to_csv("data_final.csv", index=False, sep=',')
    return data

#함수 기능: 관련 통계 데이터에 대한 새 열 이름 생성
#입력 매개변수: 열: 특성 데이터, 통계: 특성 데이터에 대해 계산할 통계
def generate_columns_name(columns, statistics):
    columns_list = []
    for column in columns:
        for statistic in statistics:
            if column not in ['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE']:  #dataframe_statistics 테이블에 없는 열 이름 필터링
                columns_list.append(statistic + "_" + column)
    return columns_list

def load_data(PATH, stage, stage_x):
    dataframe_x, dataframe_y = load_dataframe(PATH, stage, stage_x)
    train_data = abstract_statistics(dataframe_x, dataframe_y)
    train_data = train_data[train_data.columns[2:]].values         # WAFER_ID 및 STAGE 데이터 삭제
    return train_data

#함수 기능：모드 분할
def split_data(data, partitions=[50,100,165]):
    n = len(partitions)
    start = partitions[0]
    splited_data = []
    idx = np.where(data[:,-1]<=start)
    splited_data.append(np.squeeze(data[idx,:],axis=0))
    for i in range(1,n):
        end = partitions[i]
        idx = np.where(data[:,-1]<=start)
        splited_data.append(np.squeeze(data[idx,:], axis=0))
        start = end
    idx = np.where(data[:,-1]>start)
    splited_data.append(np.squeeze(data[idx,:],axis=0))
    # print(splited_data)
    return splited_data

def split_data_label(data):
    x = data[:,:-1]
    y = data[:,-1]
    return x, y

if __name__ == "__main__":
    # 훈련 데이터 세트, 테스트 데이터 세트 및 검증 데이터 세트를 별도로 생성
    # train_data
    PATH = "C:\\Users\\OnePredict\\Desktop\\CMP\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
    stage = "training"
    stage_x = 'CMP-data\\'+stage
    train_data = load_data(PATH, stage, stage_x)
    np.save("C:\\Users\\OnePredict\\Desktop\\CMP\\2016 PHM DATA CHALLENGE CMP DATA SET\\CMP-data\\Processed data set\\train_data.npy", train_data)

    # test_data
    PATH = "C:\\Users\\OnePredict\\Desktop\\CMP\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
    stage = "test"
    stage_x = 'CMP-data\\'+stage
    test_data =load_data(PATH, stage, stage_x)
    np.save("C:\\Users\\OnePredict\\Desktop\\CMP\\2016 PHM DATA CHALLENGE CMP DATA SET\\CMP-data\\Processed data set\\test_data.npy", test_data)

    # validation_data
    # PATH = "C:\\Users\\OnePredict\\Desktop\\CMP\\2016 PHM DATA CHALLENGE CMP DATA SET\\"
    # stage = "validation"
    # stage_x = stage
    # validation_data = load_data(PATH, stage, stage_x)
    # np.save("./Processed data set/validation_data.npy", validation_data)
