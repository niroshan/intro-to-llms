# Session 10: Quiz Answers

## 1. Abstract Purpose
What is the purpose of an abstract, and what should it contain?

**Answer:** The abstract is a concise summary (150-200 words) of the entire project. It should contain:
- What the project is about (topic)
- What you built/did (methods)
- What you found (key results)
- What it means (conclusion)

A reader should be able to understand your entire project from just the abstract. It's like a movie trailer for your research.

## 2. Introduction vs Background
What's the difference between the Introduction and Background sections?

**Answer:**
- **Introduction**: Explains the "why" – motivates the problem, states your goals, and previews your approach. It's about your project specifically and what you're trying to accomplish.

- **Background**: Explains the "what" – covers the technical concepts someone needs to understand your work. It's about the field and prior knowledge (what is an n-gram, what is a neural network, etc.).

The Introduction is about your project; the Background is about the concepts.

## 3. Results vs Discussion
A student puts "The RNN worked better than the bigram model because it has memory" in their Results section. Is this correct? Why or why not?

**Answer:** No, this is incorrect. 

- **Results** should present what happened (data, measurements, outputs) without interpretation. "The RNN achieved a final loss of 1.5 compared to the neural LM's 2.3" is a result.

- **Discussion** is where you interpret why things happened. "The RNN worked better because it has memory to track context" is interpretation and belongs in Discussion.

Think: Results = "what I measured"; Discussion = "what it means"

## 4. Reference Importance
Why is it important to include references in a technical report?

**Answer:**
1. **Credit**: Acknowledges the work of others that made yours possible
2. **Credibility**: Shows your work is grounded in established research
3. **Reproducibility**: Allows readers to learn more about techniques you used
4. **Academic integrity**: Avoids plagiarism by properly attributing ideas
5. **Context**: Helps readers understand where your work fits in the field

Not citing sources is academic misconduct and undermines your credibility.

## 5. Practical Task
You ran an experiment where your Transformer achieved loss of 1.2 after 500 steps, while your RNN achieved 1.5 after 500 steps. The Transformer also generated more coherent text. Write one sentence for the Results section and one sentence for the Discussion section about this.

**Answer:**
- **Results sentence**: "After 500 training steps, the Transformer achieved a final loss of 1.2, compared to 1.5 for the RNN, and produced notably more coherent multi-sentence outputs."

- **Discussion sentence**: "The Transformer's superior performance likely stems from its attention mechanism, which allows direct connections between any two positions in the sequence, enabling it to maintain coherence over longer spans than the RNN's sequential hidden state updates."
