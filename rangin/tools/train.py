from rangin.models import CHNet
from rangin.data import RDataset
from rangin.training import RanginTrainer
from rangin.utils.training import TrainArgs


model = CHNet(3, 8)
dataset = RDataset(root="data")
args = TrainArgs()
trainer = RanginTrainer(dataset, model, args)
trainer.train()
