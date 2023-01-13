import pynvml
import time
def choose_gpu(confirm_delay=5):
    pynvml.nvmlInit()
    # 这里的0是GPU id

    g_num = pynvml.nvmlDeviceGetCount()
    for i in range(g_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used/1024/1024
        if used < 100:
            time.sleep(confirm_delay)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = meminfo.used/1024/1024
            if used < 100:
                pynvml.nvmlShutdown()
                return i
    pynvml.nvmlShutdown()
    return -1

