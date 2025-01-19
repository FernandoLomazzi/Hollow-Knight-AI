import sys
import ctypes
import cv2
import numpy as np
from torchvision import transforms
import torch
import modelos
import time
import threading
import queue
import keyboard

f = open('log.txt', 'w')
#           left    right    jump    attack    dash    spell    focus
actions = ['left', 'right', 'z'   , 'x'     , 'c'   , 'd'    , 'space']


def error_handler(message):
    f.write(f"!!! ERROR !!! -> {message}\n")
    f.close()
    sys.exit(-1)


def worker_function(q):
    prev_mask = [False] * 7
    while True:
        mask = q.get()
        if mask is None:
            break
        for i, a in enumerate(actions):
            if mask[i] and not prev_mask[i]:
                keyboard.press(a)
            elif not mask[i] and prev_mask[i]:
                keyboard.release(a)
        prev_mask = mask
    for i, a in enumerate(actions):
        if prev_mask[i]:
            keyboard.release(a)


def pipe_handle():
    global f
    PIPE_ACCESS_INBOUND = 0x00000001
    FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
    PIPE_TYPE_MESSAGE = 0x00000004
    PIPE_READMODE_MESSAGE = 0x00000002
    PIPE_WAIT = 0x00000000
    PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
    ERROR_PIPE_CONNECTED = 535

    if len(sys.argv) != 1:
        error_handler("Invalid number of arguments")

    h_event = ctypes.windll.kernel32.OpenEventW(2, False, "MiEventoHKAgent")
    if not h_event:
        error_handler("Error while opening the event")
    
    pipe_name = "\\\\.\\pipe\\agent_processing"
    f.write(f"--- CREATING PIPE IN {pipe_name} ---\n")
    
    pipe = ctypes.windll.kernel32.CreateNamedPipeW(
        pipe_name,
        PIPE_ACCESS_INBOUND | FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        1, 0, 0, 0, None
    )
    ctypes.windll.kernel32.SetEvent(h_event)
    if pipe == -1:
        error_handler("S.O. error while creating pipe")
    
    f.write("--- PIPE WAITING FOR CONNECTION ---\n")
    ret = ctypes.windll.kernel32.ConnectNamedPipe(pipe, None)
    if not ret or ctypes.windll.kernel32.GetLastError() == ERROR_PIPE_CONNECTED:
        error_handler("The client connected before the pipe was available to support the connection")
    f.write("--- CLIENT CONNECTED SUCCESSFULLY ---\n")
    return pipe


def process(pipe):
    global f
    window = None
    BUFFER_SIZE = 262144

    key_queue = queue.Queue()

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    best_model_w = torch.load(f'./results/modelR5-FT-5.pt', weights_only=True)
    best_model = modelos.ResnetModel(512, 256, 2, 7, lstm_dropout=0.2, bi=False).to(device)
    best_model.load_state_dict(best_model_w)
    best_model.eval()
    try:
        while True:
            message = ctypes.create_string_buffer(BUFFER_SIZE)
            dwRead = ctypes.c_longlong()
            ret = ctypes.windll.kernel32.ReadFile(pipe, message, 1, ctypes.byref(dwRead), None)
            if not ret:
                error_handler("Error while reading, 1")
            if dwRead.value != 1:
                error_handler("Invalid message size")
            message_type = ctypes.cast(message, ctypes.POINTER(ctypes.c_longlong)).contents.value

            # Start agent message
            if message_type == 3:
                window = torch.zeros([3, 512], dtype=torch.float32, device=device)
                key_queue = queue.Queue()
                worker_thread = threading.Thread(target=worker_function, args=(key_queue,))
                worker_thread.start()
            # Frame message
            elif message_type == 12:  
                ret = ctypes.windll.kernel32.ReadFile(pipe, message, BUFFER_SIZE, ctypes.byref(dwRead), None)
                if not ret:
                    error_handler("Error while reading, 3")

                message_aux = np.frombuffer(message[:dwRead.value], dtype=np.uint8)
                img = cv2.imdecode(message_aux, cv2.IMREAD_COLOR)
                if img is None:
                    error_handler("Error while decoding image")
                img = preprocess(cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)).to(device)

                with torch.no_grad():
                    window[2] = best_model.encode(img)
                    yp = torch.nn.Sigmoid()(best_model.predict_from_encoding(window)) >= 0.5
                    key_queue.put(yp.tolist())
                window[0] = window[1]
                window[1] = window[2]
            # End agent message
            elif message_type == 48:  
                key_queue.put(None)
            # Close message
            elif message_type == 192:  
                key_queue.put(None)
                break
            else:
                error_handler("Invalid message code")
    except Exception as e:
        error_handler(e)
    ctypes.windll.kernel32.CloseHandle(pipe)


if __name__ == "__main__":
    pipe = pipe_handle()
    torch.compile(process, mode="max-autotune")
    torch.backends.cudnn.benchmark = True
    process(pipe)