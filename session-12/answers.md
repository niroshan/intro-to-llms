# Session 12: Quiz Answers

## 1. Report Structure
Put these report sections in the correct order:
Discussion, Abstract, Methods, Conclusion, Results, Introduction, Background

**Answer:**
1. Abstract
2. Introduction
3. Background
4. Methods
5. Results
6. Discussion
7. Conclusion

## 2. Figure Quality
What's wrong with this figure description: "Figure 1 shows a graph"?

**Answer:** The description is useless! A good figure caption should:
- State what the figure shows specifically ("Training and validation loss over 100 epochs")
- Mention key takeaways ("Validation loss plateaus after epoch 60")
- Provide context ("Lower loss indicates better model fit")

Better: "Figure 1: Training and validation loss curves for the RNN language model. Training loss (blue) decreases steadily while validation loss (orange) plateaus after epoch 60, suggesting onset of overfitting."

## 3. Citation Check
Which of these statements needs a citation?
a) "Our model achieved a perplexity of 45"
b) "The Transformer architecture uses multi-head attention"
c) "I found attention mechanisms fascinating"
d) "Training took approximately 2 hours"

**Answer:** Only **(b)** needs a citation. 
- (a) is your own result
- (b) is a technical fact from the literature (cite Vaswani et al.)
- (c) is personal opinion
- (d) is your own observation

## 4. Conclusion Critique
What's wrong with this conclusion?
> "In conclusion, we built models. Bigrams worked. Neural nets worked. RNNs worked. Transformers worked. The end."

**Answer:** Multiple problems:
- No synthesis (just a list, no interpretation)
- "Worked" is vague (how well? compared to what?)
- No key insights or learning
- No limitations acknowledged
- No future work suggested
- No broader significance
- Sounds unfinished and unprofessional

## 5. Reflection
List three things you learned in this course that you didn't know before Session 1.

**Answer:** (Personal – varies by student. Examples:)
1. How neural networks actually learn through backpropagation and gradient descent
2. Why attention mechanisms are so powerful for handling long sequences
3. That modern LLMs like GPT are built on relatively simple principles scaled up
4. How to implement a working language model from scratch
5. The difference between RNNs and Transformers and why Transformers dominate

## 6. Future Directions
If you had another 6 sessions, what would you want to learn/build next?

**Answer:** (Personal – valid answers might include:)
- Train on a much larger dataset (full Wikipedia, large book corpus)
- Implement proper byte-pair encoding tokenization
- Add more Transformer layers and compare scaling
- Fine-tune a pre-trained model for a specific task
- Build a simple chatbot or Q&A system
- Explore image or audio generation models
- Learn about reinforcement learning from human feedback (RLHF)
- Study how to make models safer and more aligned
