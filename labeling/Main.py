from Csv2Folder import Csv2Folder, Cor_Csv2Folder

# Vott를 활용해 라벨링한 결과 csv 파일을 폴더로 분류
sample_path = './sample/'
save_path = './class/Csv2Folder/'
# Csv2Folder(sample_path, save_path)

# 예측 결과 맞음인 이미지만 폴더로 분류
result_root_path = './../'
result_csv_path = 'submission/A_submission0.csv' # csv 파일 경로
Cor_Csv2Folder(result_root_path, result_csv_path)