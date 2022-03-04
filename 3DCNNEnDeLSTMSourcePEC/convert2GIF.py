import imageio
def compose_gif():
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure()
    canvas = FigureCanvas(fig)
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    img_paths = ["img/1.jpg","img/2.jpg","img/3.jpg","img/4.jpg"
    ,"img/5.jpg","img/6.jpg"]
    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    imageio.mimsave("test.gif",gif_images,fps=1)



filenames = []         # 存储所需要读取的图片名称
for i in range(100):   # 读取100张图片
    filename = path    # path是图片所在文件，最后filename的名字必须是存在的图片
    filenames.append(filename)              # 将使用的读取图片汇总
frames = []
for image_name in filenames:                # 索引各自目录
    im = Image.open(image_name)             # 将图片打开，本文图片读取的结果是RGBA格式，如果直接读取的RGB则不需要下面那一步
    im = im.convert("RGB")                  # 通过convert将RGBA格式转化为RGB格式，以便后续处理
    im = np.array(im)                       # im还不是数组格式，通过此方法将im转化为数组
    frames.append(im)                       # 批量化
writeGif(outfilename, frames, duration=0.1, subRectangles=False)