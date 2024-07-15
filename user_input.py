import streamlit as st

def user_input_parameters():
    with st.sidebar:
        st.markdown("<h1 style='color: black;'>Images</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #F8BBD0;'>Descriptor</h3>", unsafe_allow_html=True)
        descriptor = st.radio("Choose a descriptor", ["GLCM", "Bitdesc", "GLCM+Bitdesc"], index=0)
        st.markdown("<h3 style='color: #F8BBD0;'>Distance</h3>", unsafe_allow_html=True)
        distance = st.radio("Choose a distance", ["Manhattan", "Euclidean", "Chebyshev", "Canberra"], index=0)
        number = st.slider("Enter the number of images to display", min_value=1, max_value=10, value=5)

        return {
            "Descriptor": descriptor,
            "Distance": distance,
            "Number": number
        }
