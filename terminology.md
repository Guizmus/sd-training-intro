Term | Meaning
-- | --
Bias | Tendency of a model to give out specific features without prompting for them. For example a woman in a Disney model has a "princess" bias. Some can be worked on, but some are inherent to the chosen concept.
Bleeding | When your concept shows unprompted, it has "bled" into the rest of the model. It needs more Regularisation data.
Caption | A list of tokens to describe a picture. Usually stored in a text file named the same as the picture, or put as the filename of the picture.
Concept | Anything that can be taught through training.Examples : "a car", "My painting style", "blurry pictures", "my mother", ...
Checkpoint | See Model
Class | Pictures of other things that are more generic than your main concept. For example pictures of random cats if I was training mainly on a specific cat. Those will still get trained on too.
Concept | Anything that can be taught through training. Examples : "a car", "My painting style", "blurry pictures", "my mother", ...
Dataset | A set of data (here pictures) including Class, Regularisation and/or Instance data.
Diffusers | Diffusers are what is inside a model. The collection of weights is split between different files in different folders, such as "VAE" or "Text Encoder". You can convert from Diffusers to Models or Models to Diffusers
Duplication | A given feature that repeats in a dataset. For example the same background, or lighting, or piece of clothing.
Embeddings | Files that are used in addition to a model. They represent weights trained using Text Inversion or Lora (current methods using this file format), and apply their content to the model when they are needed. In short, using tokens that were trained on will activate the embedding, and use what was learned, as if you had changed model.
Epoch | An epoch is training on your whole dataset 1 time per repeat.
Fine-tuning | Modifying a model. Changing the weights inside a model to make it be able to understand a new concept or more. Same meaning as Training.
Full Fine-Tuning | Altering the whole model on almost every concept it understands.
Instance | Pictures of the main concept you are training. For example pictures of "Souni" my cat.
Learning rate | The learning rate is the speed at witch the model is trained, it represents how much the weights are able to change per step.
Loss | Mathematical value representing how close your model is to making pictures of the dataset. The lower the value, the best it performs.
Model | This is the big file, used by Stable Diffusion, that is the base of how an image is made. It an weight from 2GB up to 12GB currently. It's a collection of weights that represents what the AI "knows". Can be found as .ckpt or .safetensors
Overtrained | A model is considered overtrained when the pictures it makes are almost copies of the dataset pictures. It can also start to show burned outlines on the subjects.
Regularisation | Pictures of other things you don't need training specificaly. Those will still get trained on too.
Repeat | A repeat is how many times a picture is taught to the model in a given epoch.
Step | A step is training on a batch of picture one time.
Style | Any concept that conditions how an image look, how the subject in that image is represented. Examples : "My painting style", "blurry pictures"
Subject | Any concept that is the main thing represented by an image. Examples : "a car", "my mother", ...
Token | Common sequences of characters found in text. Usually about 3/4th of a word, they are the parts that constitue your prompt, and the keys to accessing the weights you are training. Choosing a token is choosing a word to train the model on.
Weights | Mathematical numbers that represents what an AI's model has retained. Those numbers are what drives SD into making its choices while making pictures and following a prompt.
