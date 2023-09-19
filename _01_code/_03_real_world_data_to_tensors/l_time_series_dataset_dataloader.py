import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class BikesDataset(Dataset):
  def __init__(self):
    bikes_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "e_time-series-bike-sharing-dataset", "hour-fixed.csv")
    # 데이터 파일 경로 지정
    bikes_numpy = np.loadtxt(
      fname=bikes_path, dtype=np.float32, delimiter=",", skiprows=1,
      converters={
        1: lambda x: float(x[8:10])  # 2011-01-07 --> 07 --> 7
      } # loadtxt 함수를 이용해 데이터파일을 numpy배열로 로드한다
      #converters 매개변수를 사용하여 데이터의 두 번째 열을 변환한다. -> 날짜 문자열을 일 정보만 추출하여 부동소수저므로 변환한다.
    )
    bikes = torch.from_numpy(bikes_numpy) # numpy 배열을 텐서로 생성

    daily_bikes = bikes.view(-1, 24, bikes.shape[1])  # -1,24,17 -> 730,24,17
    daily_bikes_target = daily_bikes[:, :, -1].unsqueeze(dim=-1) # 타겟 데이터 추출

    daily_bikes_data = daily_bikes[:, :, :-1] #
    eye_matrix = torch.eye(4)

    day_data_torch_list = []
    for daily_idx in range(daily_bikes_data.shape[0]):  # range(730)
      day = daily_bikes_data[daily_idx]  # day.shape: [24, 17]
      weather_onehot = eye_matrix[day[:, 9].long() - 1]
      day_data_torch = torch.cat(tensors=(day, weather_onehot), dim=1)  # day_torch.shape: [24, 21]
      day_data_torch_list.append(day_data_torch)

    daily_bikes_data = torch.stack(day_data_torch_list, dim=0)

    daily_bikes_data = torch.cat(
      [daily_bikes_data[:, :, :9], daily_bikes_data[:, :, 10:]], dim=2
    )

    temperatures = daily_bikes_data[:, :, 9]
    daily_bikes_data[:, :, 9] = \
      (daily_bikes_data[:, :, 9] - torch.mean(temperatures)) / torch.std(temperatures)

    self.daily_bikes_data = daily_bikes_data.transpose(1, 2)
    self.daily_bikes_target = daily_bikes_target.transpose(1, 2)

    assert len(self.daily_bikes_data) == len(self.daily_bikes_target)

  def __len__(self):
    return len(self.daily_bikes_data)

  def __getitem__(self, idx):
    bike_feature = self.daily_bikes_data[idx]
    bike_target = self.daily_bikes_target[idx]
    return {'input': bike_feature, 'target': bike_target}

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.daily_bikes_data), self.daily_bikes_data.shape, self.daily_bikes_target.shape
    )
    return str


if __name__ == "__main__":
  bikes_dataset = BikesDataset()
  print(bikes_dataset)

  print("#" * 50, 1)

  for idx, sample in enumerate(bikes_dataset):
    print("{0} - {1}: {2}".format(idx, sample['input'].shape, sample['target'].shape))

  train_dataset, validation_dataset, test_dataset = random_split(bikes_dataset, [0.7, 0.2, 0.1])

  print("#" * 50, 2)

  print(len(train_dataset), len(validation_dataset), len(test_dataset))

  print("#" * 50, 3)

  train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
  )

  for idx, batch in enumerate(train_data_loader):
    print("{0} - {1}: {2}".format(idx, batch['input'].shape, batch['target'].shape))
