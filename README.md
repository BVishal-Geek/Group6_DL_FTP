# Deep Fake Video classification

## Problem Selection & Justification
The exponential growth of multimedia content has revolutionized how we communicate and share information. Approximately 3.2 billion images and 720,000 hours of videos are shared [daily](https://www.qut.edu.au/insights/business/3.2-billion-images-and-720000-hours-of-video-are-shared-online-daily.-can-you-sort-real-from-fake). However, this surge in content has also given rise to challenges such as misinformation and deepfake manipulation.

Deepfakes—synthetically generated or altered videos and images—pose a significant threat to digital trust. These manipulations have been used for:
- Political Propaganda: Fake videos of politicians making inflammatory statements.
- Financial Fraud: Identity theft through morphed images.
- Misinformation Campaigns: Spreading false narratives with convincing fake visuals.  
The societal impact of deepfakes is profound, ranging from eroding public trust to causing financial losses and reputational damage. Detecting these manipulations is crucial to mitigating their harmful effects.
Our project aims to address this issue by developing robust deep-learning models for forgery detection. By leveraging state-of-the-art datasets like CelebDF V2 and implementing advanced architectures such as GRU, and VGG. We strive to create a reliable system for identifying manipulated multimedia content.

## Dataset
[**Celeb-DF V2**](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)
- **Description**: The Celeb-DF V2 dataset contains both real and fake (deepfake) videos of celebrities. There are a little under 1,000 real videos and over 5,000 fake videos. 

## Installing Dependencies
Install the required Python packages with:  
```pip install -r requirements.txt```

## Deep Learning Network Architecture
- CNN + GRU
- CNN + Dense


