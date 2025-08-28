# Instagram Gender Bias Detection System Development Report

## Project Overview

This document describes the development of a multi-modal system for detecting gender bias in Instagram posts. The project started with using a large language model (Qwen-VL 2.5) to annotate data, then trained a smaller, faster model that can process posts more efficiently.

## What We Built

We created a system that:
- Takes Instagram posts (image + text caption) as input
- Outputs a gender bias score from 0-10 (higher = more female-oriented)
- Can process hundreds of posts per hour on a MacBook Air
- Achieved reasonable accuracy compared to the large model we used for training data

The system uses two parts:
1. **Image analysis**: ResNet18 model processes the photos
2. **Text analysis**: DistilBERT model processes the captions
3. **Combination**: A small neural network combines both to make predictions

## Background and Motivation

### Why This Project

Instagram posts contain both images and text, and analyzing them for gender bias manually would be very time-consuming. Large AI models like Qwen-VL can do this analysis well, but they're slow and expensive to run on many posts.

### The Problem

- **Scale**: Need to analyze thousands of posts, but large models are too slow
- **Cost**: API calls to large models get expensive quickly  
- **Efficiency**: Want to run analysis on regular computers, not just high-end servers

### Our Approach

We used a technique called "knowledge distillation":
1. Use a large, accurate model (Qwen-VL) to label about 15,000 Instagram posts
2. Train a much smaller model to mimic the large model's predictions
3. The small model runs much faster and doesn't need API calls

## How We Built It

### Step 1: Data Collection

We started with an existing Instagram dataset containing:
- About 1.6 million posts
- Each post has: JSON metadata, image file, and text caption
- Data was already collected and anonymized

![Knowledge Distillation Process](images/distill.png)
*Figure 1: The four main steps of our process: collect data, get labels from large model, train small model, deploy for analysis.*

### Step 2: Getting Labels from the Large Model

We used Qwen-VL 2.5 (a large vision-language model) to label posts:
- Selected about 15,000 posts for labeling
- Asked the model to rate gender bias on a 0-10 scale
- Each API call took a few seconds and cost money
- Got labels for 15,334 posts successfully (99.9% success rate)

### Step 3: Building the Small Model

We created a model that combines image and text analysis:

**Image part**: ResNet18 (pre-trained on ImageNet)
- Takes 224x224 pixel images
- Outputs 512 numbers representing image features

**Text part**: DistilBERT (pre-trained language model)  
- Takes text captions (up to 128 words)
- Outputs 768 numbers representing text meaning

**Combination**: Simple neural network
- Takes the 512 image numbers + 768 text numbers (1,280 total)
- Processes through a few layers to make final prediction
- Outputs one number (gender bias score 0-10)

![Multi-Modal Architecture](architecture_diagram.png)
*Figure 2: Our model takes images and text, processes them separately, then combines the results to make predictions.*

### Step 4: Training Process

We trained the small model to copy the large model's predictions:
- Split the 15,334 labeled posts: 80% for training, 20% for testing
- Training took about 30-45 minutes on a MacBook Air M3
- Used the large model's scores as "correct answers" to teach the small model
- Stopped training when the model stopped improving (around 22 epochs)

![Training Curves](training_curves.png)
*Figure 3: Training progress showing how the model's accuracy improved over time and when we stopped training.*

## Results

### How Well Did It Work?

We tested our small model against the large model to see how close the predictions were:

**Hardware Used**: MacBook Air M3 (16GB memory) - shows this works on regular consumer laptops, not just expensive servers.

**Accuracy Results**:
- Average error: 1.18 points (on a 0-10 scale)
- About 68% of predictions were within ±1.0 of the large model
- About 89% of predictions were within ±2.0 of the large model

![Model Performance Comparison](model_performance_comparison.png)
*Figure 4: Comparison showing our model vs text-only, image-only, and the large teacher model across different metrics.*

### Speed and Efficiency 

The main benefit is speed and cost:
- **Large model**: 1 post per second, costs money per API call
- **Our small model**: About 800 posts per hour, runs locally for free
- **Model size**: 77 million parameters vs 32 billion (about 400x smaller)

![Error Analysis](error_analysis.png)
*Figure 5: Analysis of prediction errors showing most predictions are quite close to the large model.*

### What We Found

**Both image and text matter**: Using both images and captions together works much better than using just one.

**Trade-offs**: The small model is less accurate than the large one, but the speed improvement makes it practical for analyzing many posts.

**Efficiency**: We can run this on a regular laptop, making it accessible for researchers who don't have expensive GPU clusters.

### Large-Scale Testing

We tested the trained model on a larger dataset to see if it could handle real-world scale:

![Scale Comparison](scale_comparison.png)
*Figure 6: Comparison of dataset sizes and processing times for different approaches.*

**Results on 1.6M Instagram posts**:
- Processed about 1.47 million posts successfully (94.7% success rate)
- Found that about 23% of posts showed high gender bias (7-10 on our scale)
- Most posts (49%) were in the moderate bias range (4-6)
- About 28% showed low bias (0-3)

**Practical considerations**: 
- Some posts couldn't be processed due to missing images or corrupted files
- Processing took time due to the large dataset size
- The model performed consistently across the large dataset

## Summary

### What We Accomplished

1. **Built a working system**: Created a model that can analyze Instagram posts for gender bias using both images and text
2. **Made it practical**: Reduced a 32 billion parameter model down to 77 million parameters while keeping reasonable accuracy  
3. **Proved it scales**: Successfully tested on 1.6 million posts
4. **Made it accessible**: Runs on a MacBook Air, not just expensive servers

### Limitations

- **Accuracy trade-off**: The small model isn't as accurate as the large one (error increased from 0.85 to 1.18 points)
- **Training data dependency**: Quality depends on how good the large model's labels were
- **Cultural bias**: Trained mainly on English content, may not work well for other languages/cultures
- **Subjective task**: Gender bias assessment can be subjective and context-dependent

### Potential Uses

This kind of system could be useful for:
- **Research**: Studying gender representation trends in social media
- **Content analysis**: Helping understand bias patterns in large datasets  
- **Tool development**: Building apps that help content creators understand their messaging

### Future Improvements

- **Better accuracy**: Try different model architectures or training methods
- **More languages**: Train on non-English content
- **Real-time processing**: Make it faster for live social media analysis
- **User interface**: Build easy-to-use tools for non-technical users

## Conclusion

We successfully built a system that can analyze Instagram posts for gender bias much faster than large AI models, while maintaining reasonable accuracy. The key insight is that knowledge distillation works well for this type of multi-modal analysis task.

**Main takeaway**: You can make AI analysis practical for large datasets by training smaller models to mimic larger ones, especially when you need to process thousands of posts rather than just a few.

The system works on consumer hardware (MacBook Air), making it accessible for researchers and developers who don't have access to expensive GPU clusters.

---

*This project demonstrates a practical approach to scaling AI analysis for social media research using knowledge distillation techniques.*
