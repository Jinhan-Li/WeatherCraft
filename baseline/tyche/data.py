from torch.utils.data import Dataset
from weathercraft.data.daily import field_dataset


class DailyPairs(Dataset):
    def __init__(self, config, split):
        self.days = field_dataset(config, split)
        assert (self.days.day_ids[1:] - self.days.day_ids[:-1] == 1).all()

    def __len__(self):
        return len(self.days) - 1

    def __getitem__(self, i):
        return self.days[i]["field"][0], self.days[i + 1]["field"][0]
