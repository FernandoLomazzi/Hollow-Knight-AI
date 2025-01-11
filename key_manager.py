import pyautogui
import time
import cv2
import torch
import modelos
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actions = ['move_left', 'move_right', 'jump', 'attack', 'dash', 'spell', 'focus']

best_model_w = torch.load(f'./results/Fer 2025-01-10 16h42m18s_model_0_hidden_size=128, num_layers=2, num_classes=7, learning_rate=0.001, weight_decay=0.0, lstm_dropout=0.2, bi=False.pt', weights_only=True)
best_model = modelos.ResnetModel(512, 128, 2, 7, lstm_dropout=0.2, bi=False).to(device)
best_model.load_state_dict(best_model_w)
best_model.eval()

video = cv2.VideoCapture('test.mp4')
window = []
print_window = []

preprocess = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

for i in range(3):
	_, frame = video.read()
	print_window.append(frame)
	frame = preprocess(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)))
	window.append(best_model.encode(frame.to(device)))

count = 0
while True:
	count += 1
	ret, frame = video.read()
	if not ret:
		break
	window.pop(0)
	print_window.pop(0)

	print_window.append(frame)
	frame = preprocess(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)))
	window.append(best_model.encode(frame.to(device)))

	yp = torch.nn.Sigmoid()(best_model.predict_from_encoding(torch.stack(window).to(device)))

	if count == 150:
		stacked_frame = np.hstack(print_window)
		plt.figure(figsize=(15, 6))
		plt.title(yp.tolist())
		plt.imshow(stacked_frame)
		plt.axis('off')
		plt.show()
		break