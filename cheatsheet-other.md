# TODO: To-Be-Named

## Computer Vision

Computer vision refers to understanding images/videos, usually using ML/AI. There are many tasks of interest:
  
- Image classification: Cat vs. Dog?
- Object localization: Where are the people in this image?
- Image segmentation: What are the various parts of this image?
- Motion Detection: What moved between frames of a video?
- Etc.

### Feature importances for computer vision

- LIME can be used to shade the image based on importance
- It takes longer to render the explanation than it does the prediction.

### Neural Networks (Neural Net)

A neural network is a model similar to a pipeline. It involves a series of transformations ("layers") internally and the output is the prediction. Deep learning is using neural networks.

_**Advantages:**_

- Can learn very complex functions
  - The fundamental tradeoff is primarily controlled by the number of layers and layer sizes.
  - More layers / bigger layers -> more complex model.
  - You can generally get a model that will not underfit.
- Works really well for structured data
  - 1D sequence, e.g. timeseries, language
  - 2D image
  - 3D image or video
- Incredible successes in the last 10 years
- Transfer learning is really useful

_**Disadvantages:**_

- Often requires a lot of data
- Require a lot of compute time, and, to be faster, specialized hardware called GPUs
- Huge numbers of hyperparameters are a huge pain to tune; also slow
- Not interpretable
- Calling `fit` does not guarantee optimality
  - There are now a bunch of hyperparameters specific to fit, rather than the model.
  - You never really know if fit was successful or not.
  - You never really know if you should have run fit for longer.
- Not recommended training neural nets without further training

#### Neural Networks for Images

Two ways of processing images is:

- Flattening images (Naive)
  - Throws away a lot of useful information; computer only sees an array of numbers
- Convolutional neural networks: Can take in images without flattening them

#### Using Pre-Trained Networks

`tf.keras` has a bunch of pre-trained computer vision models. Using a dataset, we can get predictions without "doing ML ourselves". This can be useful for sentiment analysis.

_**Advantages:**_

- Saves time/cost/effort/resources
- Can use pre-trained networks directly or use them as feature transformers

#### Transfer Learning

Transfer learning is using a model trained on one task as a starting point for learning to perform another task. This is useful because it is difficult to obtain labelled data.

- Requires setting things up the same way they were set up when the model was trained (e.g., image size)

## Communication

_**Why should I care about effective communication?**_

- Most ML practitioners work in an organization with >1 people.
- There will very likely be stakeholders other than yourself who need to understand what you're doing:
  - their state of mind may change the way you do things (see below)
  - your state of mind may change the way they do things (interpreting your results)
- ML suffers from some particular communication issues
  - overstating one's results / unable to articulate the limitations
  - unable to explain the predictions
  - difficult to explain topics:
    - Why did CatBoost make that prediction?
    - Can we trust test error?
    - What does it mean if predict_proba outputs 0.9?
    - Etc.

### Principles of Good Explanations

#### Concepts then labels, not the other way around

The effectiveness of of starting with a concept then label vs the other way around varies on your audience.

- For people new to the topic, it is better to build knowledge then add label.
- For people who are familiar to the topic, it is better to label then build knowledge.

#### Bottom-up explanations

The curse of knowledge is a cognitive bias that as you learn more about a topic, it leads to top-down explanations which makes it harder to explain.
To counter this, start from an example/analogy. When you're brand new to a concept, you benefit from analogies, concrete examples and familiar patterns.

#### New ideas in small chunks

It can be beneficial to break down an explanation into small chunks

- Analogy > Example > Problem > Solution > How it works (High level) > How it works (Written Example) > How it works (Code) > Label

#### Approach from all angles

When we're trying to draw mental boundaries around a concept, it's helpful to see examples on all sides of those boundaries. If we were writing a longer explanation, it might be better to show more such as:

- Performance with and without hyperparameter tuning
- Other types of hyperparameter tuning (e.g. RandomizedSearchCV)

#### Reuse your running examples

Effective explanations often use the same example throughout the text and code. This helps readers follow the line of reasoning.

#### When experimenting, show the results asap

The first explanation shows the output of the code, whereas the second does not. This is easy to do and makes a big difference.

#### Interesting to you != useful to the reader

Communication is not only about you.

### ML and Decision-Making

There is often a wide gap between what people care about and what ML can do. To understand what ML can do, let's think about what decisions will be made using ML.

_**What is involved in decisions?**_

- **Decision variable**: The variable that is manipulated through the decision (action that will be taken).
  - EX. How much should I sell my house for? (numeric)
  - EX. Should I sell my house? (categorical)
- **Decision-maker's objectives**: The variables that the decision-maker ultimately cares about, and wishes to manipulate indirectly though the decision variable.
  - EX. Total profit, time to sale, etc.
- **Context**: The varaibles that mediate the relationship between the decision variable and the objectives.
  - EX. Housing market, cost of marketing, my timeline, etc.

_**How does this inform you as a ML practitioner?**_

Questions you have to answer:

- Who is the decision maker?
- What are their objectives?
- What are their alternatives?
- What is their context?
- What data do I need?

### Visualizing Your Results

When visualizing your results, avoid the following:

- Misleading axes
  - e.g., multiple axes, different sized axes, range of axes
- Manipulating bin sizes
  - e.g., different sized bins
- Dataviz ducks
  - e.g., tried to be clever about design but hard to read
- Glass slippers
  - e.g., applying a nice design that doesn't fit the data
- The principle of proportional ink
  - e.g., if the number is bigger, there should be more ink on the page

## [Ethics](https://github.com/UBC-CS/cpsc330/blob/master/lectures/22_ethics.ipynb)

Problems include ethics, bias, fairness, AI safety, privacy, etc.

### Bias

A model's confidence should not equate to your confidence.

_**How could this sort of bias affect peoples' lives negatively?**_
Many ways that it could affect us but unsure what exactly. Some things include advertisements, admissions, etc.

_**Where does bias come from?**_

- Data (if data only has certain groups in certain situations)
- Labels (if they were generated by humans, or not)
- Learning method (this is harder to get at)
- Bias in the way ML method is used/deployed.

### AI saftey and adversarial examples

_**Is it safe to use ML?**_
Not necessarily. An attacker can introduce noise to an image (software attack) or add a sticker with confusing material (physical attack) which can lead to a wrong prediction. With this fragility, we need to consider the application of this technology (e.g., self-driving cars). For instance, what happens if a car cannot detect a stop sign because of the sticker and runs into a person. It's not a question of do they work but rather can someone mess it up.

### Fake news and deepfakes

Fake pictures and videos and the inability to tell if it is real.

### Environmental impact

Current methods require a lot of data, time to train, many training runs to do hyperparameter optimization, resulting in a lot of energy emissions. Is this ethical?

### Crime machine learning

Predicting whether someone is a criminal based on their face

- Your prediction algorithm is only as good (or bad) as your training data
- Sources of bias:
  - Wearing a white shirt and jacket vs. other clothes
  - Facial expressions (smiling vs. frowning)
  - Cropping, lighting
  - Biased criminal justice system (e.g., tends to convict less attractive faces)
  - How can our algorithm be "better" (less biased) than humans if humans labeled the data?

### Avoiding bias in experimental set-up

- Are my results too good to be true?
- Use baselines like DummyClassifier and DummyRegressor.
- Look at feature importances.
- Manually look at some of the correct/incorrect predictions (or very low/high error for regression).
- Try making changes or perturbations (e.g. different train/validation folds) and check if your results are robust.
- When you are done, think carefully about your confidence (or credence, see lecture 21) regarding any claims you make.

### Avoiding ethical/fairness issues

- Bias usually comes from the data, not the algorithm. Think carefully about how the training data were collected.
- Familiarize yourself with how your model will be used.
- Ask yourself who might be affected by this model in deployment
