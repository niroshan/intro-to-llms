6-Week Program: Building a Mini LLM from Scratch (for a GCSE-Level Student)
Introduction

This 6-week program will guide a GCSE-level student through the fundamentals of AI and large language models (LLMs) by building a simple language model from scratch. The student will spend 1 hour every Saturday and Sunday (2 hours per week) on structured, hands-on tasks. Each week introduces key concepts (with minimal jargon) and coding exercises, gradually progressing from basic tools to a working “mini-LLM.” Along the way, the student will learn the theory and math behind language models (including neural networks, recurrent networks, and attention mechanisms) and practice explaining how LLMs work in detail. By the end of Week 6, the student will have: (a) a working miniature language model that can generate text, and (b) a written blog post/report (to be placed in their GitHub repository) summarizing the project and demonstrating their understanding.

Prerequisites & Tools: Basic Python programming (GCSE-level) and comfort with GCSE/A-level mathematics (algebra, basic calculus). We will use fundamental tools (starting with Python/Numpy and gradually introducing a library like PyTorch for efficiency) – ensuring the student learns from first principles as much as possible. The program “holds the student’s hand” with guided exercises and explanations, but encourages applying knowledge independently in later weeks.

Week 1: Introduction to Language Models and N-Gram Text Generation

Goal: Grasp what a language model is and build a simple text generator using basic probabilistic modeling (no neural networks yet). This establishes foundational understanding of how predicting the next word/character in a sequence works.

Saturday (1 hour): Learn what Language Models do. Start with a plain-language explanation: a language model is an AI system that estimates the probability of a token (word or character) or sequence of tokens appearing in a given context. In practice, this means given some text, the model predicts what comes next – enabling applications like text generation, translation, or summarization. Discuss examples (e.g. how your phone suggests next words). Introduce the concept of context: the idea that surrounding words help predict the next word. Emphasize that early language models used N-grams – sequences of N words – to capture context. (For example, a bigram model looks at the previous 1 word; a trigram looks at the previous 2 words, etc.) Longer N-grams give more context but suffer from sparsity (many possible sequences are never seen in training).

Sunday (1 hour): Build a simple N-gram text generator. Choose a small text corpus (e.g. a few chapters of a public-domain book or a Wikipedia article). Use Python to construct an N-gram frequency table: for simplicity, start with bigrams (N=2). This means counting how often each word follows each other word in the corpus. Then use this table to generate random text: pick a start word and repeatedly sample the next word based on the learned probabilities. (For example, if “the” is followed by “cat” 50% of the time and “dog” 50% in your data, then after “the” your model will choose “cat” or “dog” with equal probability.) Observe the output – it will be somewhat grammatical in short phrases but likely loses coherence beyond a few words, due to the limited context length. Discuss limitations: the bigram model doesn’t remember anything beyond one word of context, causing inconsistent or nonsensical longer sentences. This exercise illustrates the basic idea of a language model (predicting next tokens) and why we need more powerful methods to handle longer context and generalize better (N-gram models face data sparsity and can’t capture long-term structure). The takeaway: Next week, we’ll begin designing a model that “learns” patterns from data rather than relying only on observed word frequencies.

Week 2: Fundamentals of Neural Networks and Word Representations

Goal: Understand how a neural network can learn to predict text, laying the groundwork for building an AI-based language model. By the end of this week, the student will train a simple neural network that given one word (or character) predicts the next one, improving on the static N-gram approach.

Saturday (1 hour): Neural Network Basics – from Perceptron to Prediction. Introduce the concept of a neural network as a function with learnable weights that can approximate relationships in data. Start simple: a single “neuron” (perceptron) that takes inputs 
𝑥
x and produces an output 
𝑦
y via weighted sum and activation function. Explain that training such a model means adjusting the weights so that the output matches the expected result for many examples. This is done via gradient descent: the network makes a prediction, we compute a loss (error), and then we tweak the weights to reduce the error, repeating over lots of examples. Emphasize that this learning process is iterative and uses calculus (the gradient of the loss with respect to each weight) – but the student doesn’t need to derive gradients by hand for large networks; libraries can do automatic differentiation. To solidify these ideas, do a mini coding exercise: implement a tiny neural network for a very simple task such as predicting a number or classifying a small dataset (e.g. fit a line to some points, or classify flowers from sepal measurements if familiar). Use plain Python or Numpy to show how weights update. For example, show that if a model’s prediction is too high, the weight will be adjusted downward (and vice versa) – illustrating learning. Ensure the student understands terms like weights, loss function, learning rate, and epoch on a conceptual level.

Sunday (1 hour): From Words to Vectors – Neural Language Modeling. Explain how we can apply neural networks to language. Words (or characters) must be converted to numeric form to be inputs to a network. The simplest approach is one-hot encoding: represent each word in the vocabulary as a vector with a 1 in one position and 0 everywhere else. For instance, if our vocabulary is {“cat”, “dog”, “the”, “a”}, then “dog” could be [0,1,0,0]. One-hot vectors are high-dimensional and sparse, so introduce the idea of word embeddings: dense vectors that represent words in a continuous space where similar words have similar vectors. (Mention that later large models use embeddings of hundreds of dimensions where, for example, “cat” ends up near “dog” or “kitten” in that vector space.) Now, design a simple neural network for next-word prediction: input = one-hot vector of the current word, output = probabilities for the next word. Use one hidden layer (or even no hidden layer to start, i.e. a simple softmax regression) for simplicity. For example, architecture could be: one-hot input -> fully connected layer -> softmax output over vocabulary. Guide the student through coding this in Python. For a small vocabulary, you can implement training manually (looping over examples, computing predictions and updating weights via the gradient formula). However, it may be more efficient to use a framework like PyTorch at this stage: for instance, define an nn.Linear layer mapping input size (V) to output size (V), and train with cross-entropy loss. Train the model on the corpus from Week 1 (or a subset if it’s large). After training, test the neural network: give it a prompt word and see which word it predicts next (and sample a continuation). It should capture common word associations (similar to the bigram frequencies), but now the model has learned these from data rather than us explicitly programming the counts. Highlight the benefit: the neural network can generalize and adjust its predictions even for word combinations it hasn’t seen frequently (by exploiting learned weights), whereas a pure N-gram model would just have zero counts for unseen combinations. By the end of Week 2, the student should be comfortable with the idea that neural networks “learn” from examples by tuning weights (via backpropagation) to minimize prediction error, and that we can represent words numerically to feed into such networks.

Week 3: Sequence Modeling with Recurrent Neural Networks (RNNs)

Goal: Tackle the challenge of predicting sequences of multiple words/characters, not just one-step-ahead with one word of context. The student will learn how recurrent neural networks maintain a memory of past inputs and will implement a simple RNN to generate text character-by-character.

Saturday (1 hour): Introducing Recurrent Neural Networks. Discuss why the previous model (which looked at only 1 word of context) is insufficient for generating coherent sentences. We need a model that can handle sequential data and remember earlier tokens beyond a fixed window. Enter RNNs: a Recurrent Neural Network processes one item (word or character) at a time and maintains an internal hidden state that carries information from previous time steps. Explain the RNN mechanism in simple terms: at each step, the RNN takes the current input 
𝑥
𝑡
x
t
	​

 and the last hidden state 
ℎ
𝑡
−
1
h
t−1
	​

 to compute a new hidden state 
ℎ
𝑡
h
t
	​

, and possibly an output 
𝑦
𝑡
y
t
	​

. A basic formula (for a simple “vanilla” RNN) is:

ℎ
𝑡
=
tanh
⁡
(
𝑊
ℎ
ℎ
⋅
ℎ
𝑡
−
1
+
𝑊
𝑥
ℎ
⋅
𝑥
𝑡
)
h
t
	​

=tanh(W
hh
	​

⋅h
t−1
	​

+W
xh
	​

⋅x
t
	​

)

𝑦
𝑡
=
𝑊
ℎ
𝑦
⋅
ℎ
𝑡
y
t
	​

=W
hy
	​

⋅h
t
	​


where 
𝑊
𝑥
ℎ
,
𝑊
ℎ
ℎ
,
𝑊
ℎ
𝑦
W
xh
	​

,W
hh
	​

,W
hy
	​

 are weight matrices, and 
ℎ
0
h
0
	​

 (initial state) can be a zero vector. Walk through this step-by-step with a toy example (e.g. a short sequence “H-E-L-L-O” if doing characters). Show how the hidden state updates as we process each letter. The key idea: the hidden state acts like the RNN’s “memory”, allowing information from earlier in the sequence to influence later outputs. This is how an RNN can, for instance, remember that we started a sentence with “Once upon a” and thus keep generating a fairy-tale style continuation. Have the student implement a basic RNN forward pass in code for a small sequence to reinforce understanding – for example, manually compute the hidden states and outputs for a short sequence using random initial weights. (This can be done with Numpy, using small matrix multiplies for clarity.)

Sunday (1 hour): Train a Character-Level RNN Language Model. Using the concepts above, the student will build and train a simple RNN to generate text one character at a time. Character-level modeling is chosen because it dramatically reduces the vocabulary (e.g. 26 letters + punctuation), making it easier to train a model on a laptop. Use a sample text (possibly the same corpus as before, or something like an excerpt from Shakespeare for fun). Each training example will be a sequence of characters, and at each step the RNN will predict the next character. Leverage a library (PyTorch recommended) to avoid writing backpropagation from scratch for the RNN, since calculating gradients through time can be complex. For example, use torch.nn.RNN or torch.nn.LSTM (with one hidden layer) and train it on sequences from the text. (If using PyTorch’s built-in RNN, you can treat it as a black box for now – the focus is on conceptual understanding, but implementing a full backprop through time manually is not necessary given time constraints.) Monitor the training loss over iterations to see it decreasing as the model learns. After training for an hour (you might not reach perfection, but it should learn some patterns), generate some text with the RNN: start with a prompt (or even just a start character) and have the RNN produce the next character repeatedly to build a sequence. The output might be random-looking at first, but with luck and enough training, it will start to produce plausible gibberish words or even some real words reminiscent of the training data. (For example, Karpathy famously demonstrated char-RNNs can learn English-like output: “We’ll train RNNs to generate text character by character and ponder the question ‘how is that even possible?’” – showing that even a simple RNN can capture surprising structure from data.) After this session, the student should appreciate that an RNN is essentially a neural network that learns to “predict the next token given the entire history so far”, by virtue of its recurrent state. They should be able to explain how the RNN’s hidden state stores information over time (e.g. it might learn a pattern like “after the sequence ‘q-u’, likely next is ‘e’”). Discuss limitations experienced: vanilla RNNs can have trouble with very long-term dependencies (due to fading gradients) and might require tweaks or more advanced architectures to improve.

Week 4: Enhancing the Model – LSTMs and Attention Mechanisms

Goal: Build on the basic RNN by introducing two key improvements used in modern LLMs: (1) the Long Short-Term Memory (LSTM) network (or a similar gated RNN) which handles long sequences better, and (2) the concept of Attention, which later leads into Transformers. The student will experiment with an LSTM (or GRU) in code and gain a conceptual understanding of how attention allows models to focus on relevant parts of the input.

Saturday (1 hour): From RNN to LSTM – remembering longer sequences. Explain that one solution to vanilla RNN’s limitations is the LSTM, a type of recurrent network with gating mechanisms (input, output, and forget gates) that regulate the flow of information. In simple terms, an LSTM decides what to keep in “long-term memory” and what to throw away at each step, enabling it to preserve information over dozens of time steps. You can use an analogy: the LSTM has a cell state (the long-term memory) and gates that act like valves, learning to open or close to let information through. While the full LSTM equations might be too much detail, convey that LSTMs were a breakthrough that allowed RNNs to handle much longer texts than before. If time permits, have the student replace the RNN model from Week 3 with an LSTM (most deep learning libraries make this easy – e.g. nn.LSTM in PyTorch can be used similarly to nn.RNN). Train the character-level model again using the LSTM and compare: Does it learn faster or produce more coherent text than the vanilla RNN? (Often, LSTMs do achieve better results on longer sequences.) Even if the difference is small on a tiny experiment, the student should understand conceptually that LSTMs/GRUs are more powerful sequence models than basic RNNs.

Sunday (1 hour): Introduction to Attention. Begin by posing a question: When generating a word in a sentence, do we really need to remember everything that came before, or can we “attend” specifically to the relevant parts? This leads to attention mechanisms. Explain attention with a simple scenario: suppose we have a sequence and the model is trying to decide the next word – attention allows the model to look back at all prior words and assign different “weights” to each, depending on how relevant they are to the next word. It’s like having a smart highlighter over the input sequence. A brief example: in translating a sentence, to decide the translation of a particular word, an attention-enabled model can directly focus on the corresponding word in the source sentence rather than relying on a compressed fixed state. Formally, attention provides a way to compute a weighted sum of all previous hidden states, where the weights (importance) are learned via comparing queries and keys (you can mention this if the student is comfortable with linear algebra). The key point: attention allows the model to consider all positions in the sequence at once, rather than just the last hidden state. Have the student implement a toy attention example to make it concrete: e.g. take a very short sequence of values and a query vector, compute attention weights (perhaps using dot products to measure similarity) and produce an attended sum. This could be a manual calculation to show how one element “pays attention” to others. Then, explain that the groundbreaking paper “Attention Is All You Need” (2017) built an entire model (the Transformer) using attention mechanisms and no RNN recurrence. Highlight why this was huge: without sequential recurrence, Transformers can process many tokens in parallel, and handle very long contexts efficiently. In a Transformer (like GPT models), each layer can attend to all words in the input up to a certain length, learning long-range dependencies that RNNs struggled with. Describe at a high level the Transformer architecture: it’s basically a stack of layers, each layer has a self-attention head (or multiple heads) and some feed-forward neural network, allowing the model to incorporate information from all positions. (You can mention that self-attention means the model is attending to other words in the same sentence – e.g. paying attention to subject and verb to decide on agreement.) If time allows, demonstrate using an existing small Transformer model: for instance, use PyTorch’s nn.Transformer module or a mini Transformer implementation to generate text, or even just load a tiny pre-trained model (like a distilled GPT-2) to show how well it can generate text compared to our small RNN. (This step is optional and depends on computing resources; the main aim is concept over coding.) By the end of Week 4, the student should be able to explain in their own words: what an LSTM is and why gating helps, and what attention is and why Transformers replaced RNNs in modern LLMs. They should see that our “mini-LLM” so far (the char-level RNN/LSTM) is a small stepping stone toward the Transformers that power GPT-style large language models, which use attention to handle whole sequences simultaneously.

Week 5: Building a “Mini GPT” and Drafting the Report

Goal: This week, the student will consolidate their knowledge by attempting to create a simplified Transformer-based language model (if feasible), and importantly, begin writing the blog post / report that documents the project. The focus is on synthesizing what they’ve learned and demonstrating it.

Saturday (1 hour): Create a Mini Transformer Model (Optional Advanced Exercise). If the student is comfortable and time permits, guide them to implement or utilize a small Transformer for language modeling. One approach is to adapt an open-source minimal example – for instance, Karpathy’s nanoGPT – but on a much smaller scale. A simplified plan: use a very low-dimensional Transformer (e.g. 2 attention heads, 2 layers, small embedding size) and train it on the same corpus (or even a smaller one due to resource limits). Using PyTorch, this could mean defining an nn.Transformer or nn.TransformerEncoder architecture with a causal mask (so it predicts forward in text). The student should not get lost in coding details; reuse available components if possible. The learning outcome here is to recognize the structure: e.g. an embedding layer for tokens, positional encoding, then a few self-attention layers, and an output layer producing probabilities for next token. Training this mini-Transformer for a short time might not yield very coherent text (transformers usually need a lot of data and training), so manage expectations. Even if the training isn’t fully successful, the exercise is valuable for understanding how modern LLMs are set up. (If this seems too daunting, an alternative is to skip building a Transformer and instead skip to the report – the Transformer concepts from Week 4 can be discussed in the write-up without a full implementation.)

Sunday (1 hour): Begin Drafting the Blog Post/Report. Now the student switches to communication mode: documenting what they did and what they learned. Encourage them to outline the report similar to the structure of this program: introduction, week-by-week sections, and a conclusion. Key elements to include: Project overview (purpose – e.g. “I built a simple language model to learn how AI text generation works”), Methodology (what they did each week: N-gram model, neural network, RNN, etc., including any challenges and interesting results), Theory (explain in their own words core concepts like how a language model predicts the next word, how training adjusts weights, how an RNN carries state, and how Transformers use attention), and Results (describe or show samples from the mini-LLM they built – for example, include a snippet of text generated by their char-level model, and discuss its coherence). If possible, they should also mention applications or implications: e.g. “Large language models like GPT-3 are basically much bigger versions of the Transformer I explored, with billions of parameters, trained on huge text datasets.” Ensure they write in a way a layperson or a recruiter could follow, demonstrating they truly understand the material (since the goal is to show initiative and knowledge to others). By the end of this session, the student should have a rough draft of the report. It can be in Markdown (suitable for GitHub Pages or a repository README) or any shareable format. Save this draft in their GitHub repository.

Week 6: Project Completion – Refinements, Evaluation, and Final Report

Goal: Finalize the mini-LLM project and produce a polished report/blog post. This week is about review and polish: improving the model if possible, evaluating what was learned, and making the written report presentable.

Saturday (1 hour): Refine and Evaluate the Model. If the Transformer from Week 5 was implemented, evaluate its output or training progress – even if it’s not great, note any patterns (did it learn basic structure or not?). If the Transformer approach was skipped, focus on the RNN/LSTM model built earlier: maybe try training it a bit more or tuning some hyperparameters (e.g. increase the number of hidden units or training epochs slightly) to see if output quality improves. Have the student perform an evaluation of their model’s output. For instance, they can generate multiple text samples from the model given different prompts. They should analyze these qualitatively: do the outputs make sense? are there real words or any learned style mimicking the input text? Discuss why the model might still produce nonsense (e.g. small model size, limited training data). This exercise helps the student understand the limitations of their mini-LLM relative to true large-scale LLMs. It also provides material to add to the report’s results section. If any improvements were made (say, the LSTM version clearly outperforms the vanilla RNN in coherence), note that for the report. Finally, ensure all code is cleaned up and pushed to a GitHub repository. Ideally, the repo contains the training code (perhaps in a Jupyter notebook or Python scripts) and maybe some sample output files. This will allow the student to show their work to others or even reproduce results.

Sunday (1 hour): Finalize the Blog Post/Report and Publish. Have the student edit the draft from Week 5. Focus on clarity and completeness: it should explain the project from start to finish. Encourage the student to add small illustrations or diagrams if helpful (e.g. a simple flow chart of an RNN vs a Transformer encoder-decoder, or a graph of training loss over epochs if they recorded that). They should also insert citations or links for any definitions or facts – for example, if they mention “the Transformer was introduced in 2017 by Vaswani et al.” they might cite the paper or a wiki reference. Since this report may be read by others in a work experience context, ensure the tone is enthusiastic and reflective: e.g. “This project taught me how modern AI models like GPT actually work under the hood. I started with a simple word frequency model and ended with a mini version of a language model. I learned about neural networks adjusting weights through training, how recurrent networks remember information over time, and how the attention mechanism enables Transformers to consider all words at once.” Once the write-up is polished and proofread, the student should publish it to GitHub – this could be as a markdown file in a repository (for example, README.md in the project repo) or as an article on GitHub Pages or a personal blog linked in the repo. Make sure the final report highlights their understanding: someone reading it should feel the student can explain how LLMs work at a detailed level, including the math intuition (like “the model’s weights are learned via backpropagation to minimize prediction error” and “Transformers use self-attention to weigh the relevance of different words in a sentence” in the student’s own phrasing).

Wrap-Up: By the end of Week 6, the student will have a completed mini-LLM project to showcase. The deliverables are: the working code for the language model (even if rudimentary) and a well-written blog post/report. The student will have acquired foundational skills in AI: understanding how language data is processed, how a neural network learns from data, and how advanced concepts like RNNs and Transformers build on those basics to achieve the impressive feats of modern LLMs. This experience will prepare the student to explain AI concepts during work experience or interviews, demonstrating both practical initiative (coding a mini model) and theoretical knowledge (the “why” and “how” of LLMs). The structured timeline also shows the student’s ability to learn and achieve a complex task in a systematic way – a great point to discuss in any future academic or job context. Good luck, and enjoy the journey of building a mini language model! 🚀

Sources: The plan draws on established AI literature and tutorials for conceptual accuracy. For instance, the definition of language models and the progression from N-grams to neural networks is aligned with Google’s ML crash course. The description of training neural nets via weight updates is based on standard machine learning principles. Explanations of word embeddings reference the idea that words are represented in high-dimensional vector space (e.g. “cat” being near “dog”). The RNN text-generation experiment is inspired by Karpathy’s char-RNN work, demonstrating how an RNN can learn to generate text character by character. Finally, the introduction of Transformers is summarized from the seminal concept that Transformers use self-attention instead of recurrence to handle long-range dependencies and parallelize training, as well as illustrative resources on transformer architecture. These references can be cited in the student’s report to lend credibility and show engagement with the broader AI community’s knowledge.