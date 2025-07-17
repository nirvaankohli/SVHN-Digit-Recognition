import cv2
import numpy as np
import streamlit as st
import easyocr
from imutils.object_detection import non_max_suppression

st.set_page_config(
    
    page_title="House Number Detector", 
    
    layout="wide"
    
    )

# Cache one EasyOCR Reader instance (CPU mode)
@st.cache_resource

def load_reader():

    return easyocr.Reader(
        
        ['en'], 
        
        gpu=False
        
        )

def detect_number_regions(
        
        image: np.ndarray, 
        
        min_conf: float = 0.6
        
        ):
    
    """

    Run EasyOCR on the full image (RGB), allow only digits, 
    then return tight axis-aligned bboxes for anything recognized.
    
    """

    reader = load_reader()
    rgb = cv2.cvtColor(
        
        image, 

        cv2.COLOR_BGR2RGB
        
        )
    
    # detect+recognize; restrict to digits
    results = reader.readtext(

        rgb,

        allowlist='0123456789',
        
        detail=1,
                        # returns (bbox, text, confidence)
        paragraph=False

    )
    
    # collect only high-confidence pure-digit results
    boxes = []

    for bbox, text, conf in results:

        if text.isdigit() and conf >= min_conf:

            pts = np.array(bbox).reshape(-1,2)

            x1, y1 = pts[:,0].min(), pts[:,1].min()
            x2, y2 = pts[:,0].max(), pts[:,1].max()
            
            boxes.append(

                (
                    
                    int(x1), 
                    
                    int(y1), 

                    int(x2-x1), 

                    int(y2-y1)
                    
                    )
                
                )
    
    # suppress any heavy overlaps
    if boxes:
        
        rects = np.array(
            
            [[x, y, x+w, y+h] for (x,y,w,h) in boxes]

            )
        
        pick = non_max_suppression(
            
            rects, 
            
            probs=None, 

            overlapThresh=0.3
            
            )
        
        boxes = [(x1, y1, x2-x1, y2-y1) for (x1,y1,x2,y2) in pick]
    
    return boxes

def draw_boxes(
        
        image: np.ndarray, 

        boxes
        
        ):

    out = image.copy()

    for (
        
        x,

        y,

        w,
        
        h
        
        ) in boxes:

        cv2.rectangle(
            
            out, 
            
            (
                x,
                
                y

            ), 
            
            (
                
                x+w, 

                y+h

            ),

            (

                0,
                
                255,
                
                0

            ), 

            2

            )

    return out

def main():

    st.title(
        
        "🏠 House Number Detector"
        
        )

    st.write("""
             
        Upload a photo of a house (e.g. from Street View) and this app
        will use EasyOCR to find any number regions with >60% confidence.
        Adjust the slider if you need higher or lower thresholds.
    
             """)

    # slider to tweak confidence threshold
    min_conf = st.slider(
        "Min OCR confidence", 0.1, 0.99, 0.6, 0.05)
    
    uploaded = st.file_uploader("Choose a JPG/PNG image", type=["jpg","jpeg","png"])
    if not uploaded:
        return

    img_arr = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.spinner("Running OCR-based detection…"):
        boxes = detect_number_regions(img, min_conf=min_conf)
        out = draw_boxes(img, boxes)

    st.subheader(f"Detected Regions ({len(boxes)})")
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

    if boxes:
        st.success("Done! You can adjust the confidence slider to reduce false positives or catch fainter digits.")
    else:
        st.warning("No numbers found. Try lowering the confidence threshold or using a clearer image.")

if __name__ == "__main__":
    main()
