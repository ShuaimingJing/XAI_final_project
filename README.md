# XAI_final_project -- Saliency Maps Tutorial
Link for the video presentation: https://youtu.be/kmGbi9Sstrc

## Overview
This project demonstrates how to generate and analyze saliency maps for image classification models using deep learning techniques. Saliency maps highlight the important regions of an image that contribute most to a model's prediction, enabling interpretability and insights into the model's decision-making process.

The project supports four pre-trained models: ResNet50, VGG16, EfficientNet80, and MobileNet, allows users to upload or provide paths to images, generated the saliency map for each model, and integrates OpenAI's API for further analysis of generated saliency maps.

## Motivation
I am currently doing a research assiant in medical imaging field. The results of deep learning models are often used to make medical diagnoses, but it is important to understand how the models make their decisions and what they are looking for in the images. Saliency maps provide a way to understand how the model is making its decisions and what it is looking for in the images.


## How to run the project locally
#### Environment Requirements
- Python 3.8+
- Required dependencies (see requirements.txt)

#### Installation Steps
After you fork and git clone the project, You should do the following steps:
1. Prepare for the virtual environment `python -m venv venv`
2. Activate virtual environment.<br/> Windows:`venv\Scripts\activate`, MacOS or Linux:`source venv/bin/activate`
3. Install required packages `pip install -r requirements.txt`
4. Set up OpenAI API key:
- Obtain an API key from OpenAI. You can sign up for a free account at https://platform.openai.com/account/api-keys.
- Simply repalce 'your-api-key' or add it to your environment variables:
```
export OPENAI_API_KEY="your-api-key"
```

## What is salience map?
A saliency map is an interpretability tool used in deep learning, particularly for image classification models, to highlight which pixels or regions of an image are most important for the model's prediction. It visualizes the gradient of the model’s output with respect to the input image, showing how much each pixel influenced the classification decision.

In other words, a saliency map indicates the areas of focus in an image that the model relies on when making its prediction. Brighter regions in the map correspond to pixels that had a higher impact on the prediction, while darker regions had less influence.

## Results & Conclusion
This project demonstrats how to generate and analyze saliency maps for image classification models using deep learning techniques. It allows four pre-trained models: ResNet50, VGG16, EfficientNet80, and MobileNet, and integrated OpenAI's API for further analysis of generated saliency maps. Comparsion results of different models' saliency maps: ResNet50 and VGG16 have higher performance in the classification tasks among four models, and VGG16 has slight higher performance than ResNet50. According to the saliency maps, VGG16 focuses more on the relevant regions of the image rather than the background, which is more suitable for medical image analysis. (The countour lines of VGG16 is more explicit than ResNet50's saliency map) EfficientNet80 has the lowest performace, often misclassifying or showing very low confidence. MobileNet performs better than EfficientNetB0 but has notable inaccuracies in two out of three cases. 