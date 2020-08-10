import imageio
 
def create_gif(image_list, gif_name):
 
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)
 
    return
 
def gif(epochs):
    image_list = ["./tree_images/iris_"+str(epoch)+".png" for epoch in range(epochs)]
    gif_name = './tree_images/iris.gif'
    create_gif(image_list, gif_name)
 
if __name__ == "__main__":
    gif(100)