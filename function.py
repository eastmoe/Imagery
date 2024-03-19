import base64
import os
import time
import json
import requests


# 函数，读取配置文件
def getConfig():
    # 打开配置文件
    f = open('config.json', 'r')

    # 读取文件内容为content
    content = f.read()

    # 用json.loads()将读取内容转化为python字典。
    dic = json.loads(content)

    # 返回取到的内容
    #return dicc['StableDifusionWebuiAPIURL']
    return dic


# 函数，测试远程API网络连通性
def is_available(url):
    try:
        requests.get(url)
        print('API连接成功！')
        return True
    except:
        print('API连接失败，请检查SD服务器的状态与配置文件config.json里的地址是否正确。')
        return False





# 函数，将文生图信息发送给API
def send_msg_txt2img(w, h, ap, np, Actlora, ActEmb, NGTlora, NGTEmb, hi_enable, hr_scale, hr_upscaler,
                     hr_second_pass_steps, cfg, step, denoise, seed, sample):
    # 组织请求信息
    payload = {
        "width": w,
        "height": h,
        "prompt": ap + ' ' + (' '.join(Actlora)) + ' ' + (' '.join(ActEmb)),
        "negative_prompt": np + ' ' + (' '.join(NGTlora)) + ' ' + (' '.join(NGTEmb)),
        "hi_enable": hi_enable,
        "hr_scale": hr_scale,
        "hr_upscaler": hr_upscaler,
        "hr_second_pass_steps": hr_second_pass_steps,
        "cfg_scale": cfg,
        "steps": step,
        "denoising_strength": denoise,
        "seed": seed,
        "sample_index": sample[0]
    }

    # 发送请求，同时获取API URL。
    config=getConfig()
    api=config["StableDifusionWebuiAPIURL"]
    response = requests.post(url=f'{api}/sdapi/v1/txt2img', json=payload, timeout=3600)
    r = response.json()

    # 确定output文件夹是否存在，不存在则创建该文件夹
    path = "output\\txt2img"
    if not os.path.exists(path):
        os.makedirs(path)
        print("output\\txt2img文件夹已经自动创建。")
    # 组合绝对路径
    path = os.path.join(os.getcwd(), path)

    # 解码并保存文件，依据当前时间为文件命名
    file_path = os.path.join(path, f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.png")
    with open(file_path, 'wb') as f:
        f.write(base64.b64decode(r['images'][0]))
    # 返回生成的图片内容：
    return file_path


# 函数，将图生图信息发送给API
def send_msg_img2img(initimage, w, h, ap, np, Actlora, ActEmb, NGTlora, NGTEmb,
                     cfg, step, denoise, seed, sample):
    # 组织请求信息
    payload = {
        "init_images": [initimage],
        "width": w,
        "height": h,
        "prompt": ap + ' ' + (' '.join(Actlora)) + ' ' + (' '.join(ActEmb)),
        "negative_prompt": np + ' ' + (' '.join(NGTlora)) + ' ' + (' '.join(NGTEmb)),
        "cfg_scale": cfg,
        "steps": step,
        "denoising_strength": denoise,
        "seed": seed,
        "sample_index": sample[0]
    }
    #print(payload)
    # 发送请求，同时获取API URL。
    configdic = getConfig()
    api = configdic["StableDifusionWebuiAPIURL"]
    response = requests.post(url=f'{api}/sdapi/v1/img2img', json=payload, timeout=3600)
    r = response.json()

    # 确定output文件夹是否存在，不存在则创建该文件夹
    path = "output/img2img"
    if not os.path.exists(path):
        os.makedirs(path)
        print("output/img2img文件夹已经自动创建。")
    # 组合绝对路径
    path = os.path.join(os.getcwd(), path)

    # 解码并保存文件，依据当前时间为文件命名
    file_path = os.path.join(path, f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.png")
    with open(file_path, 'wb') as f:
        f.write(base64.b64decode(r['images'][0]))
    # 返回包含绝对路径的文件名
    return file_path


# 函数，用于保存上传文件
def saveuploadfile(image):
    # 确定upload文件夹是否存在，不存在则创建该文件夹
    path = "upload"
    if not os.path.exists(path):
        os.makedirs(path)
        print("upload文件夹已经自动创建。")
    # 组合绝对路径
    path = os.path.join(os.getcwd(), path)
    # 构造文件路径，确保路径以斜杠结尾
    file_path = os.path.join(path, image.name)
    # 使用二进制写入模式保存图像文件
    with open(file_path, "wb") as f:
        image_contents = image.getvalue()
        f.write(image_contents)
    # 返回完整文件路径
    return file_path


# 函数，获取模型列表
def getmodel():
    # 发送请求，同时获取API URL。
    configdic=getConfig()
    api=configdic["StableDifusionWebuiAPIURL"]
    response = requests.get(url=f'{api}/sdapi/v1/sd-models', )
    r = response.json()
    # 初始化一个空列表来存储model_name
    chklist = []
    # 遍历api_response中的每个元素
    for item in r:
        # 提取model_name字段并添加到列表中
        chklist.append(item["model_name"])

    # 获取VAE列表
    response = requests.get(url=f'{api}/sdapi/v1/sd-vae', )
    r = response.json()
    # 初始化一个空列表来存储model_name
    vaelist = []
    # 遍历api_response中的每个元素
    for item in r:
        # 提取model_name字段并添加到列表中
        vaelist.append(item["model_name"])

    # 获取Lora列表
    response = requests.get(url=f'{api}/sdapi/v1/loras', )
    r = response.json()
    # 初始化一个空列表来存储model_name
    Loralist = []
    # 遍历api_response中的每个元素
    for item in r:
        # 提取model_name字段并添加到列表中
        Loralist.append(item["name"])

    # 获取embedding列表
    response = requests.get(url=f'{api}/sdapi/v1/embeddings', )
    r = response.json()
    # 初始化一个空列表来存储model_name
    emblist = []
    # 遍历api_response中的每个元素
    for item in r:
        # 提取模型文件名字段并添加到列表中
        emblist = list(r['loaded'].keys())

    # 返回获取到的当前checkpoint、vae和lora，emb列表
    return chklist, vaelist, Loralist, emblist


# 函数，将上传的图片转化为base64编码。
def encode_image(img_path):
    with open(img_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


# 函数，切换模型
def switch_checkpoint(chk, vae):
    option_payload = {
        "sd_model_checkpoint": chk[0],
        "sd_vae": vae[0]
    }
    # 获取API地址
    config=getConfig()
    api=config["StableDifusionWebuiAPIURL"]
    response = requests.post(url=f'{api}/sdapi/v1/options', json=option_payload)
    if response.status_code == 200:
        return True
    else:
        return False


# 函数，获取系统信息
def get_server_info():
    # 获取系统信息
    config=getConfig()
    api=config["StableDifusionWebuiAPIURL"]
    response = requests.get(url=f'{api}/internal/sysinfo', )
    r = response.json()
    # 使用空list存储数据
    #sysinfo=[]
    #for item in r:
        #sysinfo=list(r['Platform'].keys())
        #sysinfo = list(r['Platform'].keys())
    return r


# 函数，获取当前任务进度
def get_progress():
    # 获取API地址
    config=getConfig()
    api=config["StableDifusionWebuiAPIURL"]
    response = requests.get(url=f'{api}/sdapi/v1/progress', )
    r= response.json()
    # 获取eta_relative和progress的值
    eta_relative = r.get("eta_relative", None)
    progress = r.get("progress", None)
    return progress,eta_relative

