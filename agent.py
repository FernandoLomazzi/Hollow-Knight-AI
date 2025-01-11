from sys import exit, argv
import ctypes
from cv2 import VideoWriter, VideoWriter_fourcc, imdecode, IMREAD_COLOR, resize, cvtColor, COLOR_BGR2RGB
from numpy import array, frombuffer, uint8
import pyautogui
from torchvision import transforms
import torch
import modelos
import time
import threading
import queue

PIPE_ACCESS_INBOUND = 0x00000001
FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
PIPE_TYPE_MESSAGE = 0x00000004
PIPE_READMODE_MESSAGE = 0x00000002
PIPE_WAIT = 0x00000000
PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
ERROR_PIPE_CONNECTED = 535

# Constants
BUFFER_SIZE = 262144 #524288
WIDTH = 224
HEIGHT = 224
FOLDER_PATH = "F:/videos/test/"  # Replace with actual path
f = open('log.txt', 'w')

def error_handler(message):
    f.write(f"!!! ERROR !!! -> {message}\n")
    f.close()
    exit(-1)

def worker_function(q):
    keys_pressed = []
    while True:
        #seconds = time.time()
        keys = q.get()  # Get keys from the queue
        if keys is None:
            break
        for a in keys:
            if a not in keys_pressed:
                pyautogui.keyDown(a)
        for a in keys_pressed:
            if a not in keys:
                pyautogui.keyUp(a)
        keys_pressed = keys
        #print(time.time() - seconds)
    for a in keys_pressed:
        pyautogui.keyUp(a)


def pipe_handle():
    global f
    if len(argv) != 1:
        error_handler("Cantidad de argumentos inválida")

    h_event = ctypes.windll.kernel32.OpenEventW(2, False, "MiEventoHKAgent")
    if not h_event:
        error_handler("Error al abrir el evento.")
    
    pipe_name = "\\\\.\\pipe\\agent_processing"
    f.write(f"--- CREANDO EL PIPE EN {pipe_name} ---\n")
    
    pipe = ctypes.windll.kernel32.CreateNamedPipeW(
        pipe_name,
        PIPE_ACCESS_INBOUND | FILE_FLAG_FIRST_PIPE_INSTANCE,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
        1, 0, 0, 0, None
    )
    ctypes.windll.kernel32.SetEvent(h_event)
    if pipe == -1:
        error_handler("Error de sistema operativo en la creacion del pipe.")
    
    f.write("--- PIPE ESPERANDO CONEXION ---\n")
    ret = ctypes.windll.kernel32.ConnectNamedPipe(pipe, None)
    if not ret or ctypes.windll.kernel32.GetLastError() == ERROR_PIPE_CONNECTED:
        error_handler("El cliente se conectó antes que el pipe esté disponible para admitir la conexión. El cliente es muy rápido.")
    f.write("--- CLIENTE CONECTADO SATISFACTORIAMENTE ---\n")
    return pipe

def process(pipe):
    global f
    writer = None
    window = []
    prev_actions = []
    key_queue = queue.Queue()
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #                  ['left', 'right', 'jump', 'attack', 'dash', 'spell', 'focus']
    actions = array(['left', 'right', 'z'   , 'x'     , 'c'   , 'd'    , 'space'])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    best_model_w = torch.load(f'./results/Fer 2025-01-10 16h42m18s_model_0_hidden_size=128, num_layers=2, num_classes=7, learning_rate=0.001, weight_decay=0.0, lstm_dropout=0.2, bi=False.pt', weights_only=True)
    best_model = modelos.ResnetModel(512, 128, 2, 7, lstm_dropout=0.2, bi=False).to(device)
    best_model.load_state_dict(best_model_w)
    best_model.eval()
    play_time = 5 # 20 is a second
    f.write("a")
    try:
        count = 0
        while True:
            message = ctypes.create_string_buffer(BUFFER_SIZE)
            dwRead = ctypes.c_longlong()
            ret = ctypes.windll.kernel32.ReadFile(pipe, message, 1, ctypes.byref(dwRead), None)
            if not ret:
                error_handler("Error en la lectura. 1")
            if dwRead.value != 1:
                error_handler("Tamaño de tipo de mensaje no reconocido.")
            message_type = ctypes.cast(message, ctypes.POINTER(ctypes.c_longlong)).contents.value
            f.write(str(message_type) + "\n")

            if message_type == 3:  # New file name message
                ret = ctypes.windll.kernel32.ReadFile(pipe, message, BUFFER_SIZE, ctypes.byref(dwRead), None)
                if not ret:
                    error_handler("Error en la lectura. 2")
                #file_name = message[:dwRead.value].decode('utf-8')
                #writer = VideoWriter(FOLDER_PATH + file_name + ".mp4", 
                #                         VideoWriter_fourcc('m', 'p', '4', 'v'), 
                #                         20.0, 
                #                         (WIDTH, HEIGHT))
                #f.write(f"Video creado con éxito: {file_name}.mp4\n")
                window = []
                prev_actions = []
                key_queue = queue.Queue()
                worker_thread = threading.Thread(target=worker_function, args=(key_queue,))
                worker_thread.start()
                # q.put(None)
            elif message_type == 12:  # Frame message
                ret = ctypes.windll.kernel32.ReadFile(pipe, message, BUFFER_SIZE, ctypes.byref(dwRead), None)
                if not ret:
                    error_handler("Error en la lectura. 2")

                message_aux = frombuffer(message[:dwRead.value], dtype=uint8)
                img = imdecode(message_aux, IMREAD_COLOR)
                if img is None:
                    error_handler("Error al decodificar la imagen.")

                img = preprocess(cvtColor(resize(img, (WIDTH, HEIGHT)), COLOR_BGR2RGB))
                window.append(best_model.encode(img.to(device)))

                if len(window) == 3:
                    count += 1
                    if count & 1 == 0:
                        yp = torch.nn.Sigmoid()(best_model.predict_from_encoding(torch.stack(window).to(device))) >= 0.5
                        actions_done = actions[yp.cpu()]
                        #print(count, actions_done)
                        #count += 1
                        key_queue.put(actions_done)
                    #for a in prev_actions:
                    #    pyautogui.keyUp(a)
                    #for a in actions_done:
                    #    pyautogui.keyDown(a)
                    #pyautogui.press(actions_done)
                    #prev_actions = actions_done
                    window.pop(0)
                    #play_time -= 1
            
            elif message_type == 48:  # End video message
                #writer.release()
                #f.write("Video finalizado con éxito\n")
                window = []
                prev_actions = []
                key_queue.put(None)
            elif message_type == 192:  # Close message
                #writer.release()
                #f.write("Video finalizado con éxito. Cerrando pipe\n")
                window = []
                prev_actions = []
                key_queue.put(None)
                break        
            else:
                error_handler("Código de tipo de mensaje no reconocido.")
    except Exception as e:
        error_handler(e)
    ctypes.windll.kernel32.CloseHandle(pipe)

if __name__ == "__main__":
    pipe = pipe_handle()
    process(pipe)