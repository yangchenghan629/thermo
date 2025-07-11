from PIL import Image
gif=[]
gif2=[]
for i in range(73):
    img=Image.open(f'./skewTR/skewT-{i//3:02d}{i*20%60:02d}.png')
    gif.append(img)
gif[0].save('skewTR.gif',save_all=True,append_images=gif[1:],duration=500,loop=0)
