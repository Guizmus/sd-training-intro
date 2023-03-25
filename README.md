# Introduction

In this guide, every ***emphased word*** is defined in the [Terminology section](#terminology).

The training process for Stable Diffusion consists of multiple steps, and even if the method you choose may change, the concepts you'll encounter will remain the same.

For simple trainings, lots of those will be of low importance, but as you push further the limits, you'll want to understand each of the following principles :

* [Safety note](#safety-note)
* [Defining a concept](#defining-a-concept)
  * [What is a concept ?](#what-is-a-concept-)
  * [How many concepts can I train at once ?](#how-many-concepts-can-i-train-at-once-)
  * [Full Fine-Tuning](#full-fine-tuning)
* [Making a dataset](#making-a-dataset)
  * [Dataset size](#dataset-size)
  * [Captioning](#captioning)
  * [Attention](#attention)
  * [Regularization](#regularization)
  * [Dataset diversity](#dataset-diversity)
* [Choosing a training method](#choosing-a-training-method)
  * [Full Fine-Tuning](#full-fine-tuning-1)
  * [Dreambooth](#dreambooth)
  * [LoRA](#lora-fine-tuning-with-a-smaller-size)
  * [Hypernetworks](#hypernetwork-additional-output-conditioning)
  * [Textual Inversion](#textual-inversion-better-accessing-of-the-model-weights)
* [Monitoring the training](#monitoring-the-training)
  * [Learning Rate](#learning-rate)
  * [Training by batch](#training-by-batch)
  * [Steps/Epochs/Repeats](#stepsepochsrepeats)
  * [Saving checkpoints](#saving-checkpoints)
  * [Monitoring the Loss value](#monitoring-the-loss-value)
  * [Control pictures](#control-pictures)
* [Evaluating the training](#evaluating-the-training)
* [Terminology](#terminology)


# Safety Note

Be careful when downloading ***models/checkpoints***, you are responsible for taking safety measures for what you choose to download.

Due to the nature of open source software, security risks can occasionally arise. ***Checkpoints*** (.ckpt) can be infected with executable code, and while some hosting platforms such as Hugging Face are implementing scanners to help mitigate this potential security risk, always be sure to do your due diligence in regards to where you are getting your ***models*** from.

As a safer alternative, you could use the .safetensors format - we strongly recommend that you convert .ckpt to .safetensors.

Links to .safetensors convertors :
* [Hugging Face space](https://huggingface.co/spaces/safetensors/convert)
* [Extension in Automatic1111](https://github.com/Akegarasu/sd-webui-model-converter)

[Back to index](#introduction)

# Defining a concept

## What is a concept ?

A ***concept*** is anything you want to teach to the ***model***, anything that you have enough graphical representation to show and train the ***model*** on.

It falls under two categories : ***Subjects*** and ***Styles***.

## How many concepts can I train at once ?

There is no hard limit to that, but the more ***concepts*** you add, the harder it can get to train them all effectively, and the longer the training will take.

As ***models*** don't grow in size during training, the longer you train, the more quality will be harmed on what your model knew before the training.

Training on other things too (***regularization/class*** data) can help prevent this effect to a certain point.

## Full Fine-Tuning

Usually, you train only on a few ***concepts*** at once. ***Full Fine-Tuning***, on the contrary, aims at training the whole ***model*** on every ***concept*** it can. It requires an enormous ***dataset*** to do correctly and keep all concepts high quality.

[Back to index](#introduction)

# Making a dataset

A ***dataset*** is a collection of data (here pictures) that represent your concepts, and that will be taught to your model.

It is composed of :
* ***Instance*** data (pictures of what you want to train)
* ***Regularization*** data or Class data (pictures of diverse other things)

## Dataset size

A dataset size can greatly vary :
* training on a single ***subject*** ***concept***, you usually train between 5 and 20 pictures.
* training on a ***style*** ***concept***, you use from 50 to 200 pictures.
* ***Full Fine-Tuning*** a ***model***, you train on multiple thousands pictures. Each ***concept*** included is to be treated at least like a single ***concept*** would, with around 20 to 50 pictures each.

## Captioning

Each picture has a ***caption*** linked to it. It can be saved as the name of the file itself, or as a text file placed alongside the picture. It contains ***tokens*** that should describe the picture.

There are different approach to captioning :
* Single ***token caption***. By naming each picture with a single ***token***, the "instance prompt", you will focus the training on the ***concept***, and train faster.
* Multi ***token caption***. By describing multiple elements in the picture, you split the attention of the training, and train more concepts at once. This also helps to reduce the impact of a recurring feature in your dataset.
* Full sentence ***caption***. Using a full sentence as a ***caption*** is a particular case of multi ***token***, when you want to train on lots of ***tokens*** at once. This is mostly useful in ***Full Fine-Tuning***.

## Attention

Each ***step*** of the training, a batch of pictures is trained, and the ***weights*** of the ***model*** move a little. Those changes happen slower or faster, depending on the ***learning rate*** you use. This "budget" of changes that could happen on a single step is called ***Attention***, and is split amongst the ***tokens*** you used in your ***caption***.

Adding more ***tokens*** to a ***caption*** then has multiple effects :
* it slows down the training on each single ***token***. This may require more total steps to produce the same results, or to use more trained ***tokens*** at once in your prompt later on.
* it looks for the more fitting parts of the picture for that ***token*** and associates with it the changes. This means that describing a feature in your ***caption*** can prevent that feature from being associated with the other ***tokens***. As an example, if training on a character that has a tie in half the shots, adding the ***token*** "tie" would reduce how much of the tie feature is associated with your character.
* it spreads the training on more ***weights*** of the ***model***, and reduces the need for ***regularization***.

## Regularization

***Regularization*** data is not what you want to train on, but is a good ally to have in your ***dataset***. ***Regularization*** data should be of good picture quality since those pictures get trained on too.

Most training tools put that data in a specific folder, and ask you to give a ***token*** to train that data on. Depending on the method of training, it can be called "Class data" too. It wants a quite generic concept, and you should have lots of diverse representations of it. Usually, the ***class*** data is from 20 to 100 times the size of the ***instance*** data.

Some tools, instead of letting you use a single ***token*** for all your ***class*** data, let you caption the ***regularization*** data just as well as you caption your ***instance*** data. The same concepts regarding Attention apply then.

In both case, the goal of the ***regularization*** is to keep most of the model training, and not have the new concepts you are training currently push the model too strongly in one direction, as well as add diversity.

## Dataset diversity

Training on a ***dataset*** is asking for the recurring features to be learned and associated with your ***token***. Because of this, it's extremely important to not have unwanted recurring features.

For example, if all pictures are taken in daylight, or if half the characters wear a tie, those features will be learned. In particular, repeating motifs, or text, will get picked up very fast and can harm the quality of your training.

The good approach is to try to not have any duplication of any kind, and keep everything that shouldn't be learned changing.

This can be a difficult task, and you may need to come back to this step once you finish your training and see ***biases*** in the results

[Nitrosocke's guide](https://github.com/nitrosocke/dreambooth-training-guide/blob/main/README.md)

[Back to index](#introduction)

# Choosing a training method

The training process for Stable Diffusion offers a plethora of options, each with their own advantages and disadvantages.

Essentially, most training methods can be utilized to train a singular ***concept*** such as a ***subject*** or a ***style***, multiple ***concepts*** simultaneously, or based on ***captions*** (where each training picture is trained for multiple ***tokens***). The choice of training method will ultimately impact the quality and level of difficulty during the training process.

Training involves multiple steps. An important one is finding the right method and tool for your case.

## Full Fine-Tuning
Everydream is a powerful tool that enables you to create custom ***datasets***, preprocess them, and train Stable Diffusion models with personalized ***concepts***.

This provides a general-purpose ***fine-tuning*** codebase for Stable Diffusion ***models***, allowing you to tweak various parameters and settings for your training, such as batch size, ***learning rate***, number of ***epochs***, etc.

Moreover, it supports various types of diffusers, including DDIM, U-Net, and ResNet.

* Difficulty: medium-hard
* Style training: very good quality
* Subject training: very good quality
* GPU cost: very high (16GB or more)

[EveryDream](https://github.com/victorchall/EveryDream2trainer)

[notebook](https://colab.research.google.com/github/nawnie/EveryDream2trainer/blob/main/Train_Colab.ipynb)

[Difference : EveryDream & Dreambooth](https://github.com/victorchall/EveryDream2trainer/blob/main/doc/NOTDREAMBOOTH.md)

## Dreambooth

Dreambooth is a technique that enables you to ***fine-tune*** diffusion ***models*** by incorporating custom ***concepts*** into them.

With Dreambooth, you have the ability to take your own images and train the AI to recognize an object (person, image, or thing, etc.) and then synthesize it wherever you desire.

This process involves updating your entire ***checkpoint/model*** and reworking all the ***weights***, by teaching it some new ***tokens***, using new pictures.

It outputs a .ckpt file, or diffusers.

* Difficulty: easy (one subject) to very hard (multiple concepts/styles)
* Style training: very good quality
* Subject training: very good quality
* GPU cost: high (12GB or more)

[Original diffusers library](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)

[Extension in Automatic1111](https://domain.ext/path)

[ShivamShrirao](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth)

[notebook](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/DreamBooth_Stable_Diffusion.ipynb)

[TheLastBen](https://github.com/TheLastBen/fast-stable-diffusion)

[notebook](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb)

## LoRA: fine-tuning with a smaller size

LORA is a type of ***embedding*** training method, which functions similarly to Dreambooth.

Although it might have slightly lower quality results, it compensates with a much faster training process and smaller file sizes than Dreambooth.

You may also encounter other terminology, like LoHa and LoCon, declinations of the LoRA method, [more maths details on those here.](https://github.com/KohakuBlueleaf/LyCORIS/blob/lycoris/Algo.md)

Currently, it's regarded as one of the best options available in terms of balancing quality and requirements.

* Difficulty: easy-hard (less chance of destroying the base model)
* Style training: good quality
* Subject training: good quality
* GPU cost: medium (8GB or more)

[Original github](https://github.com/cloneofsimo/lora)

[Diffusers library](https://huggingface.co/docs/diffusers/training/lora)

[Kohya_ss library](https://github.com/kohya-ss/sd-scripts)

[notebook](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-dreambooth.ipynb )

## HyperNetwork: additional output conditioning

By utilizing hypernetworks in combination with a model, it becomes feasible to use them with numerous models with only a minimal reduction in quality. Hypernetworks work by driving the standard output towards a specific direction, thus providing more refined results. Although they are quite easy to train and accessible, it may require a significant number of steps to train them effectively.

* Difficulty: easy-medium
* Style training: good quality
* Subject training: medium quality
* GPU cost: medium (8GB or more)
* Integrated in the training tab of Automatic1111

## Textual Inversion: better accessing of the model weights
Textual Inversion is considered the weakest technique as it does not directly modify the model. Nevertheless, multiple instances of Textual Inversion can be utilized concurrently and are only activated when prompted. Additionally, they are lightweight and simple to share.

* Difficulty: easy
* Style training: good quality
* Subject training: medium quality
* GPU cost: medium (8GB or more)

[Diffusers library](https://huggingface.co/docs/diffusers/training/text_inversion)

[Integrated in the training tab of Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion)

[Back to index](#introduction)

# Monitoring the training

During the training session, there are multiple parameters that play an important role, and that are present in most tools :

## Learning Rate

The ***learning rate*** is the speed at which your model will be trained. A higher value will result in a shorter training session. It has some dangerous sides effects though :
- the training can go faster into the ***overtrained*** state.
- the training can fail completely and diverge, meaning it manages less and less to draw pictures like you wanted.

## Training by batch

Like when making pictures, you have access to the concept of Batch during training. A Batch of picture is a group of similar size pictures that will be trained on together. Having more than one single picture per batch is good for quality, but it has diminishing returns after a point.

Two main parameters let you change this :
- ***Batch Size***. This will take more VRAM, and train all the pictures in parallel, at the same time on your GPU. This is the fastest option, but more demanding on hardware.
- ***Gradiant Accumulation Steps*** (GAS for short). This will act the same, but sequentially and not in parallel, training multiple pictures one after the other before impacting the model with the changes for all at once.

Those two parameters work together, and the resulting total picture count per batch is equal to Batch Size multiplied by GAS.

To optimize speed and quality of the training, you should have your Batch size to the highest value your computer can handle, and use GAS as a multiplier to your batch size. Since your dataset will be split in batches, it's more efficient to make total picture count per batch a divider of your dataset size.

## Steps/Epochs/Repeats
Those 3 elements are linked and will represent the length of the training.
* A ***step*** is training the model on one batch of pictures one time.
* A ***repeat*** is how many times a given picture is taught to the model during an epoch.
* An ***epoch*** is training the model on the whole ***dataset*** as many times as repeats.

## Saving checkpoints
During the training, you have parameters to ask the training to save every X epochs, starting on epoch Y. It's quite important to do some saving regularly like this, in order to have a fallback when you have ***overtrained*** the model.

## Monitoring the Loss value
Most tools give access to a ***loss*** value that shows in the training console, and that gets saved in a tensorboard folder automatically. You can run open the tensorboard and indicate the directory where those logs are saved, in order to access a webUI showing graphs of the Loss value during the training.

To open a tensorboard UI, you will need to open a console, activate the python environment of your training tool and then run ```tensorboard --logdir="PATH"```, replacing the PATH with your local tensorboard path.

***Loss*** is a mathematical value that tries to represent how well your model is currently trained on your pictures. The lower it is, the closest your model will create pictures from the dataset.

There are some important behaviors to look for in this ***Loss*** graph :
- If the value goes down, and then goes up again, the lower value it attained is a possible good training result, take the ***checkpoint*** that happened the closest to it.
- If the value goes rapidly up, and doesn't seem to come down after some more epochs, going even faster up, there is a good chance your training has failed and is diverging. Reduce the Learning Rate and restart.
- If the value goes down and down faster and faster, you have reached the ***overtrained*** state. The model you have now should mostly make copies of what was in the ***dataset***.

## Control pictures

While the training is happening, you may have some samples that get created. There are usually 2 sets of pictures. One should become closer and closer to what you are training, while another one should remain fuzzy and unclear.

If the pictures that were fuzzy start to show your trained concept, then your concept is bleeding into the model, meaning it will start to show up without being prompted for. This means you need to add better regularization data.

[Back to index](#introduction)

# Evaluating the training
Once you are done training, you will have one or more ***checkpoints*** to compare, at different ***epochs*** trained.
Load those up into your UI, and use the X/Y/Z plot feature, in order to compare the quality of each ***checkpoint*** on the same prompt and seed.
Prompt all the ***tokens*** you tried to train on, rate the quality of each, try to detect any ***bias***.
Depending on those results, you may want to modify your dataset :
- If your ***concept*** does show correctly, but the face or some specific features are of poor quality, include some more close up pictures of those features.
- If your ***concept*** shows but has some strange burned outline, and feels too similar to your dataset, you have overtrained. You may want to look for duplicates in your dataset that would present the features that start to burn first, and add more variety on it. Sometimes, this means just removing a picture.
- If your ***concept*** shows unprompted, you want to add or improve your regularization data.
- If your ***concept*** has most of the time some specific feature you don't want, look for examples of this feature in the dataset. Either remove those pictures, or add another token in the caption to describe that ***bias*** and diverge the ***attention*** to it.

[Back to index](#introduction)

# Terminology

Term | Meaning
-- | --
Attention | Each step of the training, a batch of pictures is trained, and the weights of the model move a little. Those changes happen slower or faster, depending on the learning rate you use. This "budget" of changes that could happen on a single step is called Attention.
Batch Size | Number of pictures to train in parallel on your GPU. Faster training, more quality, but VRAM intensive.
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
Gradiant Accumulation Steps | Number of batches of pictures to train sequentially before impacting the model with the changes. Functionally similar to Batch size, but slower, and without VRAM impact.
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

[Back to index](#introduction)
