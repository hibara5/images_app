import streamlit as st
from user_input import user_input_parameters
from descriptor import glcm, bitdesc, glcm_bit
from distances import manhattan, euclidean, chebyshev, canberra, retrieve_similar_images
import numpy as np
import cv2
import os

# Load precomputed signatures
glcm_signatures = np.load('glcm_signatures.npy', allow_pickle=True)
bitdesc_signatures = np.load('bitdesc_signatures.npy', allow_pickle=True)
glcm_bit_signatures = np.load('glcm_bit_signatures.npy', allow_pickle=True)

descriptor_funcs = {"GLCM": glcm, "Bitdesc": bitdesc , "GLCM+Bitdesc" : glcm_bit}
distance_funcs = {"Manhattan": manhattan, "Euclidean": euclidean, "Chebyshev": chebyshev, "Canberra": canberra}

def main():
    st.set_page_config(page_title='Images App', page_icon=':camera:')
    
    st.markdown("<h1 style='text-align: center; color: #F8BBD0; font-weight: bolder;'>Images App</h1>", unsafe_allow_html=True)

    input_values = user_input_parameters()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    file = st.file_uploader("Upload your image 😊:", type=["csv", "txt", "jpg", "png"])
    
    st.write("Selected Options:")
    st.write(input_values)
    
    if file is not None:
        # Read the uploaded image in color
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
            
            st.markdown("<h3 style='text-align: center; color: #F8BBD0; font-weight: bolder;'>Uploaded Image:</h3>", unsafe_allow_html=True)
            st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
            
            descriptor_func = descriptor_funcs[input_values["Descriptor"]]
            query_features = descriptor_func(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  # Convert to grayscale for descriptor
            
            if input_values["Descriptor"] == "GLCM":
                signature_db = glcm_signatures
            elif input_values["Descriptor"] == "Bitdesc":
                signature_db = bitdesc_signatures
            else:
                signature_db = glcm_bit_signatures
            distance_func = distance_funcs[input_values["Distance"]]
            
            similar_images = retrieve_similar_images(signature_db, query_features, distance_func, input_values["Number"])
            
            st.markdown("<h3 style='text-align: center; color: #F8BBD0; font-weight: bolder;'>Result : </h3>", unsafe_allow_html=True)
            
            cols = st.columns(3) 
            for i, (img_path, dist, label) in enumerate(similar_images):
                full_img_path = os.path.join('./dataset', img_path)
                if os.path.exists(full_img_path):
                    img_similar = cv2.imread(full_img_path)
                    img_similar_rgb = cv2.cvtColor(img_similar, cv2.COLOR_BGR2RGB)
                    with cols[i % 3]: 
                        st.image(img_similar_rgb, caption=f"Image {i+1} (Distance: {dist.round(3)})")
                        st.write(f"{full_img_path}")
                else:
                    st.write(f"Image not found: {full_img_path}")
            
            st.markdown("<h3 style='text-align: center; color: #F8BBD0; font-weight: bolder;'>Thank you</h3>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: black; font-weight: bolder;'>By Hiba Rajai</h3>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
