import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io


blur = np.array([[
    [0.0625,0.125,0.0625],
    [0.125,0.25,0.125],
    [0.0625,0.125,0.0625]
]])

edge = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])
sharpen = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

v_edge = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

h_edge = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])

bblur = np.array([
    [1/9,1/9,1/9],
    [1/9,1/9,1/9],
    [1/9,1/9,1/9]
])


def convolve(img: np.array,kernel: np.array):
    
    height,width,c = img.shape[0],img.shape[1],img.shape[2]

    K_height,K_width = kernel.shape[0],kernel.shape[1]

    convolved_img = np.zeros((height-K_height+1,width-K_width+1,3))

    #with padding
    for i in range(K_height//2,height-K_height//2 -1):
        for j in range(K_width//2,width-K_width//2-1):

            subset = img[i-K_height//2:i+K_height//2+1,j-K_width//2:j+K_width//2+1]

            convolved_img[i,j,0] = int((subset[:,:,0]*kernel).sum())
            convolved_img[i,j,1] = int((subset[:,:,1]*kernel).sum())
            convolved_img[i,j,2] = int((subset[:,:,2]*kernel).sum())

    convolved_img = np.clip(convolved_img,0,255)
    return convolved_img.astype(np.uint8)

        
def show_img(img: str,kernel: str)->np.array:
    # if kernel == 'blur':
    #     kernel = blur
    # elif kernel == 'edge':
    #     kernel = edge
    # else:
    #     kernel = sharpen
    kernel_map = {
        'blur':blur,
        'BoxBlur':bblur,
        'edge':edge,
        'sharpen':sharpen,
        'VerticalEdge':v_edge,
        'HorizontalEdge':h_edge
    }
    selected_kernel = kernel_map.get(kernel,sharpen)
    img = Image.open(img)
    img = np.asarray(img)
    image = convolve(img,selected_kernel)
    # image = Image.fromarray(image)
    # image.show()
    return image
    
def main():
    st.title("Play with Kernels!")
    st.divider()
    st.subheader("Choose Image")
    data = st.file_uploader("upload an Image!")
    st.subheader("Select a kernel")
    # choice1 = st.radio("Pick kernel size",["3","5","7"])
    choice = st.selectbox("PickOne",["blur","edge","sharpen","BoxBlur","VerticalEdge","HorizontalEdge"])
    st.write("You selected kernel of size: 3 and type: ",choice)
    # st.subheader("Or take a picture")
    # image = st.camera_input("take picture")
    
    st.subheader("Output!")
    if data is not None:
        if st.button("process") is not None:
            file_bytes = np.asarray(bytearray(data.read()),dtype=np.uint8)
        # show_img(file_bytes,choice)
            image1 = cv2.imdecode(file_bytes,1)
            cv2.imwrite('image_path.jpg', image1)

        # Use the image file path as needed
            image_path = 'image_path.jpg'
            output = show_img(image_path,choice)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            _, encoded_image = cv2.imencode('.png', output_rgb)
            image_bytes = encoded_image.tobytes()

# Convert the image bytes to a BytesIO object
            image_io = io.BytesIO(image_bytes)
            
# Display the image in Streamlit
            st.image(image_io, channels="RGB")
            st.download_button("Download Image", data=image_bytes, mime="image/png")
        # st.image(output,channels="BGR")
           
        # st.pyplot(p)
        # p.savefig("output.png")
        # st.download_button("Download Image")
main()