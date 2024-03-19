import streamlit as st
from PIL import Image
import threading
from queue import Queue
import time
import function

# debug使用，输出时间
# print(f"\n启动时间：{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}")

# 获取默认配置，并赋予变量默认值
config_dict = function.getConfig()
output_file = None
i2i_img_base64 = ""
sample_meth = config_dict["sample_meth"]
api_url = config_dict["StableDifusionWebuiAPIURL"]
default_step = config_dict["Step"]
default_h = config_dict["H"]
default_w = config_dict["W"]
Ngtloraselected = config_dict["Ngtloraselected"]
Embselected = config_dict["Embselected"]
NGTEmbselected = config_dict["NGTEmbselected"]
hi_enable = config_dict["hi_enable"]
hr_scale = config_dict["hr_scale"]
hr_upscaler = config_dict["hr_upscaler"]
hr_second_pass_steps = config_dict["hr_second_pass_steps"]
seed = config_dict["seed"]
cfg = config_dict["cfg"]
disnoise = config_dict["disnoise"]
bar_value = 0
result_queue = Queue()


# 设置网页标题
#st.title('AI画图')
st.title('Imagery')

# 展示一级标题
st.header('')

# 设置折叠区域
server_info = st.expander('系统信息')
if server_info.checkbox('显示服务端信息'):
    server_info.json(function.get_server_info())

# 创建一个事件对象，用以触发子线程
event = threading.Event()

# 创建一个空进度条，表示生成进度
progress_bar = st.progress(bar_value,"进度")

# 函数，用于进度条进度更新
def update_progress_bar_status():
    # 等待1秒后开始
    time.sleep(1)
    while True:
        # 循环周期是0.5秒
        time.sleep(0.5)
        # 获取当前任务进度
        prograss_present, eta_present = function.get_progress()
        #print("\nProgress:",int(prograss_present * 100))
        # 当进度为0时，设置进度条为100%并跳出循环
        if prograss_present == 0:
            progress_bar.progress(100, '进度')
            break
        else:
            # 转换进度为当前进度百分比
            progress_bar.progress(int(prograss_present * 100), '进度')







# 获取Stable Diffusion WEBUI API端的模型信息
chk_list, vae_online, loralists, emblist = function.getmodel()

# 侧边栏
with st.sidebar:
    # checkpoint选择框
    chk = st.selectbox(
        label='CheckPoint模型：',
        options=chk_list,
        index=0,
        format_func=str,
        help='选择您需要使用的CheckPoint模型'
    )
    # vae选择框
    vae = st.selectbox(
        label='VAE：',
        # options=('橘子', '苹果', '香蕉', '草莓', '葡萄',vae_online),
        options=vae_online,
        index=0,
        format_func=str,
        help='选择您需要使用的VAE模型'
    )
    # 切换模型按钮
    if st.button('切换模型'):
        with st.spinner('切换中'):
            if function.switch_checkpoint(chk, vae):
                st.success(f'切换模型成功！')
            else:
                st.error(f'切换模型失败！')
    # 图生图
    i2i = st.checkbox('启用图生图', value=False)
    if i2i:
        # 选中选择框之后，显示上传按钮
        uploaded_image = st.file_uploader(label="选择图片文件",
                                          accept_multiple_files=False,
                                          type=["jpg", "jpeg", "png", "bmp"],
                                          help='提供一张用作起点的图像')
        disnoise = st.slider(label='重绘幅度(Denoising Strength)',
                             min_value=0.00,
                             max_value=1.00,
                             value=disnoise,
                             step=0.01,
                             help='该参数的范围从0到1。\n其中0根本不添加噪声，你将获得添加的确切图像，1完全用噪声替换图像，几乎就像你使用普通的文生图而不是图生图一样。')

    # 图片尺寸调整
    weight = st.number_input(label='宽度',
                             min_value=32,
                             max_value=2048,
                             value=default_w,
                             step=1,
                             help='请输入需要生成的图片的宽度')

    height = st.number_input(label='高度',
                             min_value=32,
                             max_value=2048,
                             value=default_h,
                             step=1,
                             help='请输入需要生成的图片的高度')

    # 步数调整滑块
    user_steps = st.slider(label='迭代步数',
                           min_value=1,
                           max_value=200,
                           value=default_step,
                           step=1,
                           help="请输入您需要迭代的步数"
                           )

    # 提示词文本框
    actpro = st.text_area(label='提示词:',
                          value='',
                          height=5,
                          max_chars=9999,
                          help='请输入正面提示词，最大长度为9999字符')
    # Lora选择框
    loraselected = st.multiselect(
        label='Lora模型选择：',
        options=loralists,
        default=None,
        format_func=str,
        help='选择您需要使用的正向Lora模型'
    )

    # 负面提示词文本框
    ngtpro = st.text_area(label='负面提示词:',
                          value='',
                          height=5,
                          max_chars=9999,
                          help='请输入负面提示词，最大长度为9999字符')
    # 高级设置
    advance = st.checkbox('启用其他设置', value=False)
    if advance:
        # 选中选择框之后，显示以下选项
        # Lora选择框
        Ngtloraselected = st.multiselect(
            label='反向Lora模型选择：',
            options=loralists,
            default=None,
            format_func=str,
            help='选择您需要使用的负面Lora模型'
        )

        # Embedding选择框
        Embselected = st.multiselect(
            label='文本转义(Embedding)模型选择：',
            options=emblist,
            default=None,
            format_func=str,
            help='选择您需要使用的正向Embedding模型'
        )
        # 负面Embedding选择框
        NGTEmbselected = st.multiselect(
            label='负面文本转义(Embedding)模型选择：',
            options=emblist,
            default=None,
            format_func=str,
            help='选择您需要使用的负向Embedding模型'
        )



        seed = st.number_input(label='随机种子',
                               min_value=-1,
                               max_value=9999999999999,
                               value=seed,
                               step=1,
                               help='随机种子是一个决定的初始随机噪声的数字。\n如果多次使用相同的提示运行相同的seed的时候，你会得到相同的生成图像。')
        cfg = st.number_input(label='提示词引导系数(CFG Scale)',
                              min_value=0,
                              max_value=30,
                              value=cfg,
                              step=1,
                              help='CFG这个参数可以看作是“创造力与提示”量表。\n较低的数字使AI有更多的自由发挥创造力，而较高的数字迫使它更多地坚持提示词的内容。\n默认的CFG是7，这在创造力和生成你想要的东西之间是最佳平衡。通常不建议低于5，因为图像可能开始看起来更像AI的幻觉，而高于16可能会开始产生带有丑陋伪影的图像。')

        sample_meth = st.selectbox(label='采样方法(Sampleer)',
                                   options=('Euler', 'Euler a', 'DPM++ 2M Karras', 'DPM++ SDE Karras', 'DDIM'),
                                   index=2,
                                   format_func=str,
                                   help='这些采样器是算法，它们在每个步骤后获取生成的图像并将其与文本提示请求的内容进行比较，然后对噪声进行一些更改，直到它逐渐达到与文本描述匹配的图像。')
        # 测试API连接
        if st.button('测试API'):
            # 显示一个旋转的加载器，表示任务正在执行。
            with st.spinner('测试中...'):
                api_status = function.is_available(url=api_url)
                if api_status:
                    st.success(f'{api_url}\n连接成功！')
                else:
                    st.error(f'{api_url}\n连接失败！')

    # 高清修复
    if i2i:
        st.info("在图生图模式下，暂不支持高清修复。")
    else:
        hi_enable = st.checkbox('启用高分辨率修复', value=False)
        if hi_enable:
            hr_scale = st.slider(label='放大倍率',
                                 min_value=1.00,
                                 max_value=4.00,
                                 value=hr_scale,
                                 step=0.02,
                                 help='将生成的图片等比例放大。')
            hr_upscaler = st.selectbox(
                label='采样器',
                options=(
                    'Latent', 'Lanczos', 'Nearest', 'DAT x4', 'ESRGAN_4x', 'LDSR', 'R-ESRGAN 4x+',
                    'R-ESRGAN 4x+ Anime6B',
                    'ScuNET GAN', 'SwinIR 4x', 'ScuNET PSNR'),
                index=0,
                format_func=str,
                help='选择您需要使用的高清修复方式')
            hr_second_pass_steps = st.number_input(label='高清修复步数',
                                                   min_value=0,
                                                   max_value=50,
                                                   value=hr_second_pass_steps,
                                                   step=1,
                                                   help='高清修复所需的额外步数')





    # 生成按钮
    if st.button('生成'):

        # 定义图生图包装函数，它使用队列来传递结果
        def img2img_pack(uploaded_image, weight,height,actpro,ngtpro,loraselected,Embselected,Ngtloraselected,NGTEmbselected,
                     cfg,user_steps,disnoise,seed,sample_meth,result_queue, event):
            # 等待事件被触发
            event.wait()
            # 调用原始函数并获取结果
            result= function.send_msg_img2img(initimage=function.encode_image(function.saveuploadfile(uploaded_image)), w=weight,h=height,ap=actpro,np=ngtpro,
                                              Actlora=loraselected,ActEmb=Embselected,NGTlora=Ngtloraselected,NGTEmb=NGTEmbselected,
                                              cfg=cfg,step=user_steps,denoise=disnoise,seed=seed,sample=sample_meth)
            # 将结果放入队列
            result_queue.put(result)

        # 定义文生图包装函数，它使用队列来传递结果
        def txt2img_pack(weight,height,actpro,ngtpro,loraselected,Embselected,Ngtloraselected,NGTEmbselected,cfg,user_steps,disnoise,seed,sample_meth,hi_enable,hr_scale,hr_upscaler,hr_second_pass_steps,result_queue, event):
            # 等待事件被触发
            event.wait()
            # 调用原始函数并获取结果
            result= function.send_msg_txt2img(w=weight,h=height,ap=actpro,np=ngtpro,
                                              Actlora=loraselected,ActEmb=Embselected,NGTlora=Ngtloraselected,NGTEmb=NGTEmbselected,
                                              cfg=cfg,step=user_steps,denoise=disnoise,seed=seed,sample=sample_meth,hi_enable=hi_enable,
                                              hr_upscaler=hr_upscaler,hr_scale=hr_scale,hr_second_pass_steps=hr_second_pass_steps)
            # 将结果放入队列
            result_queue.put(result)


        with st.spinner('生成中...'):
            # 当模式为图生图时：
            if i2i:
                # 创建图生图包装函数的子线程
                t1= threading.Thread(target=img2img_pack,args=(uploaded_image, weight,height,actpro,ngtpro,
                                                               loraselected,Embselected,Ngtloraselected,NGTEmbselected,
                                                            cfg,user_steps,disnoise,seed,sample_meth,result_queue, event))
                t1.start()
                # 在生成图像的任务完成后触发事件
                event.set()
                # 更新进度条
                update_progress_bar_status()
                # 等待子线程结束
                t1.join()
                # 从队列取出图片
                output_file = result_queue.get()
            # 当模式为文生图时：
            else:
                # 创建文生图包装函数的子线程
                t2= threading.Thread(target=txt2img_pack,args=(weight,height,actpro,ngtpro,loraselected,Embselected,
                                                               Ngtloraselected,NGTEmbselected,cfg,user_steps,disnoise,
                                                               seed,sample_meth,hi_enable,hr_scale,hr_upscaler,hr_second_pass_steps,
                                                               result_queue, event))
                t2.start()
                # 在生成图像的任务完成后触发事件
                event.set()
                # 更新进度条
                update_progress_bar_status()
                # 等待子线程结束
                t2.join()
                # 从队列取出图片
                output_file = result_queue.get()
        # st.error('错误，未开放此功能！')
        # t1.join()  # 等待进度条线程完成

# 主页面显示图生图原始图像
if i2i:
    # 上传文件不为空时
    if uploaded_image is not None:
        # 保存图片并获取路径
        i2i_upimgpath = function.saveuploadfile(image=uploaded_image)
        image = Image.open(uploaded_image)
        # 在图片区域显示图片
        st.image(image,
                 caption='图生图：输入的原图',
                 width=768,
                 )

# 显示生成结果
if output_file is not None:
    out_image = Image.open(output_file)
    st.image(out_image,
             caption='输出图',
             width=768,
             )
    # 提供下载图片按钮
    with open(output_file, "rb") as file:
        btn = st.download_button(
            label="保存生成图片",
            data=file,
            file_name="output.png",
            mime="image/png"
        )


