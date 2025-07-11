from PIL import Image
gif=[]
gif2=[]
for i in range(73):
    img=Image.open(f'./1.1/profile{i:02d}.png')
    gif.append(img)
gif[0].save('./result/1.1profile.gif',save_all=True,append_images=gif[1:],duration=200,loop=0)
