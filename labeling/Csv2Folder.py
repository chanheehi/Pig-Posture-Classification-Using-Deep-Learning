import cv2, glob, os, csv

def Csv2Folder(sample_path, save_path):
    names, labels = [], []
    # csv 파일 리스트로 저장
    with open(sample_path+'label.csv', 'r', newline='', encoding='cp949') as f:
        reader = csv.reader(f)
        for num, row in enumerate(reader):
            names.append(row[0])
            labels.append(row[1])

    # 이미지 imgs에 불러오기
    imgs = []
    for num, file_name in enumerate(names):
        img = cv2.imread(sample_path+file_name)
        imgs.append(img)

    # 폴더에 이미지 파일 저장
    for num, img in enumerate(imgs):
        cv2.imwrite(save_path+labels[num]+'/'+names[num], img)


def Cor_Csv2Folder(result_root_path, result_csv_path):
    # csv에서 라벨링 결과 맞음/틀림 불러오기
    names, pred_labels, correct_labels, result = [], [], [], []
    with open(result_root_path+result_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for num, row in enumerate(reader):
            if num == 0:
                continue    # 첫번째 줄은 라벨링 결과가 아니므로 제외
            names.append(row[0])    # 파일 이름
            pred_labels.append(row[1])   # 라벨 예측 결과
            correct_labels.append(row[2])   # 라벨 정답
            result.append(row[3])   # 맞음/틀림

    # 이미지 imgs에 불러오기
    imgs, correct_file_names, pred_file_labels, correct_file_labels  = [], [], [], []
    for num in range(len(names)):
        # 라벨이 맞음인 경우에만
        if result[num] == '맞음':
            try:
                img = cv2.imread(result_root_path+'dataset/test/'+correct_labels[num]+'/'+names[num])   # 라벨링 결과가 맞는 이미지
                imgs.append(img)    # 이미지
                correct_file_names.append(names[num])   # 파일 이름
                pred_file_labels.append(pred_labels[num])   # 라벨 예측 결과
                correct_file_labels.append(correct_labels[num]) # 라벨 정답
            except:
                print('error:',num)

    # 폴더에 이미지 파일 저장
    for num, img in enumerate(imgs):
        cv2.imwrite('./class/Cor_Csv2Folder/'+correct_file_labels[num]+'/'+correct_file_names[num], img) 
        # 파일 저장후 기존 파일 삭제
        # if os.path.isfile(result_root_path+'dataset/test/'+pred_file_labels[num]+'/'+correct_file_names[num]):
        #     os.remove(result_root_path+'dataset/test/'+pred_file_labels[num]+'/'+correct_file_names[num])