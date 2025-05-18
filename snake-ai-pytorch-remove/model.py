import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from multiprocessing import Process, Queue

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            self.eval()
        else:
            print(f"Model file {file_path} not found.")

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

def train_model(queue, model, lr, gamma, data):
    trainer = QTrainer(model, lr, gamma)
    for state, action, reward, next_state, done in data:
        trainer.train_step(state, action, reward, next_state, done)
    
    queue.put(model.state_dict())  # Отправляем модель обратно в родительский процесс

if __name__ == "__main__":
    input_size = 4  # Замените на ваш размер входа
    hidden_size = 64  # Замените на ваш размер скрытого слоя
    output_size = 2  # Замените на ваш размер выхода
    model = Linear_QNet(input_size, hidden_size, output_size)
    
    lr = 0.001
    gamma = 0.99

    # Пример данных для тренировки (замените на ваши данные)
    training_data = [
        (torch.rand(input_size), torch.tensor([1, 0]), 1.0, torch.rand(input_size), False),
        (torch.rand(input_size), torch.tensor([0, 1]), 0.5, torch.rand(input_size), True),
        # Добавьте больше данных...
    ] * 10  # Умножаем для примера

    num_processes = 4  # Количество потоков
    processes = []
    queue = Queue()
    
    for _ in range(num_processes):
        p = Process(target=train_model, args=(queue, model, lr, gamma, training_data))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Получаем финальные веса модели
    model.load_state_dict(queue.get())
    print("Training completed.")