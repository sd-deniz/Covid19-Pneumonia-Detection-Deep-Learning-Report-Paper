## Covid19 Pneumonia Detection Project by using *Artificial Intelligence* Techniques
(March, April 2021)

Aim of this study is designing a deep learning model to detect Covid 19 disease using x-ray lung images from 
[“Kaggle, CoronaHack -Chest X-Ray-Dataset”](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/metadata) dataset. 
This dataset has unlabeled CT images and labeled x-ray images. 
There are 7 different categories in total, 1 normal and 6 diseases with lung inflammation symptoms. 
Kaggle dataset was highly unbalanced. Therefore, compared to the network architecture more study was made on the efficient use of the dataset that is detected to have faulty labels.
(To test the model there is an app written in python in [**this**](https://github.com/sd-deniz/Deep-Learning-Covid19-Detection-Evaluation-App) repository.
Lung x-ray images can be tested with one line of shell command.)

*As a summary of this study and the repository,*

An accuracy of 88% and 91% was obtained in two different test sets.
The methodology used in this process can be interpreted as a test of author's mathematical intuituion about statistics and it is not enough to claim deterministic results. The major deviations in the methodology carried out in this project were mostly due to the limited computation time provided by Google Colab. These boundary conditions, by its nature, led the author to two findings: Going deeper with ResNet models does not create much of a difference at least for chest x-ray images and secondly, data-centric approach (mostly ablation and train) functioned.

As mentioned in more detail in the "General Interpretations" section, 'Texas Sharpshooter Fallacy'-esque statistical biases may occur in the process of collecting and using data for diagnosis of diseases. Therefore, it is deduced that, in order to create reliable artificial intelligence diagnosis systems, machine learning pipeline's could be used locally on the basis of institutions. Establishing an international platform for research on the determination of standards and safety protocols in the use of artificial intelligence in medicine can speed up the work and make it safer.

Overall conclusion is to find balance between neural network model, data generation & labeling and data-centric approach due to the reasons mentioned in the 'General Interpretations' section.

---

Research paper ***update***:  
At [this study](https://arxiv.org/pdf/2101.06871.pdf) made by Andrew Y. Ng et al., they tested the performances of pretrained models by training them with chest X-ray images.

The results about the model deepness and x-ray image relationship in the arXiv article supports the small-scale model comparison that is made in this repository. 

---
<p>&nbsp;</p>

### Kaggle, CoronaHack X-Ray-Dataset Distribution, *Table 1*

|Category|Number of Images|
|:----------|:--------:|
|Normal     |1342|
|Bacteria   |2530|
|Virus      |1345|
|Covid19|58 |58  |
|Sars, Virus|5   |
|Streptococcus Bacteria|4|
|Stress Smoking|2|

<p>&nbsp;</p>

### Neural Network Model

**Modified ResNet18 Model Architecture:** Model architecture of the succesfull(relatively) model


> **ResNet18 (**Pretrained=<span style="color:blue">True</span>**)** 
>>FC (512, 128) >> resnet output << 
>>>Dropout (0.5) 
>>>> FC (128,3)
>>>>> CrossEntropyLoss (log_softmax+nllloss)

<p>&nbsp;</p>


<img src="https://raw.githubusercontent.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/master/markdown_files/model_architecture_whitebackground.png" width="800" style="background-color: transparent;">

**Summary of the models**

After selecting the model structure, different training methods was used in different combination in order to see the effect of the methods.
ResNet 50 and ResNet 18 architectetures was used for pretrained model. These models were tested along with the modified resnet18 mentioned above. Every model runs with same settings. Only the output class number changes to 3 or 7.
**Result Model** , *unlike others*, was babysitted by changing training data distribution. As a result, only the babysitted "Modified ResNet18" model (Result Model) have a relatively better generalized model compare to others. The algorithm followed for babysitting while training the "Result Model" will be explained in the "Training algorithm of Result Model" section. 

All models except the "Result Model" were trained with the same parameters. These are:

- All of the training data feeded into model with the *batch_size = 128*
- Training / Validation split is 0.85/0.15
- Training / Validation split is stratified according to training set
- Batches were shuffled each epoch.
- Optimizer method: Stochastic Gradient Descent with Momentum    
- Criterion: torch.nn.CrossEntropyLoss (nn.LogSoftmax + nn.NLLLoss)


```python
optimizer = optim.SGD(params=model.parameters(), 
				      lr=0.001, 
				      momentum=0.95, 
				      weight_decay=0.0002
                      )
 
criterion = nn.CrossEntropyLoss()
```

<p>&nbsp;</p>

As seen in table image below,

Different pretrained ResNet50 and ResNet18 and Modified ResNet18 were trained **11 epochs** with 7 classes and 3 class outputs.

- 7 classes: Normal, Bacteria, Virus, Covid19, SarsVirus, Strept. Bact., Stress Smoking

- 3 classes: Normal, *Remaining Inflammations such as Bact, Virus, Stress Smoking ...* , Covid19

Purple colered cells belongs to [“Kaggle, CoronaHack -Chest X-Ray-Dataset”](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/metadata) 's Train and test set. (Note: 3 random Covid images moved into test set from train set. Which is called **Test**. Light Blue colored cells belongs to [COVID-19 & Normal Posteroanterior(PA) X-rays](https://www.kaggle.com/tarandeep97/covid19-normal-posteroanteriorpa-xrays). Whole dataset used as **Test2**. This dataset contains 140 Normal and 140 Covid Chest X-ray images. Considering the Coronahack dataset has faulty labeled images, test2 dataset was used for comparison. **Test2** dataset was used for comparison purposes after all models including Result Model were trained and the project was terminated. Test2 was not involved in the decision-making process. Test2 was seen after the model training was completed.

***Accuracy and Loss Metric:***  
*Kaggle Coronahack dataset was very small compare to the ImageNet dataset that ResNet was trained on. To be able to track and understand the dataset easily, arithmetic mean and F1 score was used simultaneously for the training, validation and test set.*
**Confusion matrix of the test results have been created in order to track F1 score.**

Loss: Each epoch,  Arithmetic mean of the Batch Losses

Accuracy: Each epoch, (number of Correct Prediction) / (number of Total Prediction). It was used for logging the results without any data loss.

 
[![](https://raw.githubusercontent.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/master/markdown_files/model_result_table.png)](https://raw.githubusercontent.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/master/markdown_files/model_result_table.png)

(During the training of Result Model, learning rate and sample distribution were actively changed depending on the loss, accuracy of training and validation dataset, overfit, plateau relationship. The momentum value was also changed depending on the batch size and sample distribution either 0.90(last 10), 0.95(last 20). Model detected 3 out of 3 covid images in Test set. This is denoted by the blue asterix.)

**Interpretation of the results**

1. None of these models except the babysitted "result model" was able to detect 3 covid images in the **Test** set. But ResNet50, **deeper model**, detected some of the covid images in the **Test2** dataset unlike others. In the table image this is denoted by (1) and (2) in orange color. And the models that couldn't detect covid images are denoted by the red asterix.

2. Before training all these models, changing class weight of ResNet50 resulted in a same situation where model was able to detect some covid images correctly. With these results, it was deduced that similar results can be obtain by changing sample size an distribution.

3. By looking at the table, the deeper ResNet model does not seem to cause any significant improvement.

The algorithm followed during the training of Result Model is formulated in section "Training algorithm of Result Model".
The results that was achieved with "human-in-the-loop" such as changing sample size by looking at metrics, could have been achieved with Class Weights. This was tested. The decision process to change the learning rate between 0.1 and 0.001 in order to find different local minimum was implemented into training as given in the below algorithm.


<p>&nbsp;</p>

**Training algorithm of Result Model**

*Start*

>**Train model**  
(if model improves save the model)

*If improves keep training.*

>If F1 score and accuracy dont increase after few epochs, decrease the learning rate.  
Train the model  
(if model improves save the model)

*If improves keep training.*

>If the model is stuck with a poor F1 score and accuracy, change the sample distribution in favor of incorrectly predicted labels.  
Train the model  
(if model improves save the model)

*If improves keep training.*

>If the model overfits and metrics drop then it might be a bad local minimum.
Go back to last saved model. Increase the learning rate between 10x ~ 100x.  
Probably, gradients will jump into a different local minimum.

> **Back to top.**

*End*

<p>&nbsp;</p>

***General Interpretations:***

It can be deduced that the factors that may cause these prediction differences may caused by the variances that medical imaging machines have, such as machine material, software, signal processesing methods et cetera. It is observed that papers which has published by known publishers has quality medical imaging datas. These quality medical imaging datas was created with more advanced machines compare to the ones of public datas. And most of the times these quality datas are not publicly available. Also private hospitals, institution are not willing to share their datas. Few institutions share their datas and when they share, these datas are being used by the researchers of top tier universities.

In addition, images may carry local variances and similarities like genetics, life styles. For instance, if there is an industrial air pollution in the area where the tests are performed, or if there is a coal-fired thermoelectric power plant nearby, the normal classified x-ray images will contain lung inflammation [(ref)](https://pubmed.ncbi.nlm.nih.gov/21623970/). And these images, most likely, will be classified as non-Covid / Normal images. These images could be helpful in order to prevent overfit but in the data collecting process valuable datas such as 'inflammation caused by air pollution' label will be lost over time. Even if this situation does not occur, it will not be possible to determine the situations that have already occurred. [(Texas sharpshooter fallacy)](https://en.wikipedia.org/wiki/Texas_sharpshooter_fallacy).
As a result, it has been concluded that installing machine learning pipelines locally(per institution) can be helpful in diagnosis, considering factors such as machines, their calibrations, operators, diagnostician.

Inflammation is an immune response and it alone is not a unique indicator for Covid19 detection. Other type of tests like blood minerals, hormons, hemograms can be taken into consideration in order to build a scalable, usefull prediction model for the future of the disease diagnosis since representative word "allostasis" take care of the body regulation. ([*allostasis*](https://en.wikipedia.org/wiki/Allostasis#cite_note-SterlingEyer1988-2): *remaining stable by being variable*).
For covid detection task, dog nose could be a alternative option with high success rates that is proven and currently used in airports. Perhaps the combination of [biohybrid nose](https://singularityhub.com/2021/01/26/scientists-made-a-biohybrid-nose-using-cells-from-mosquitoes/) and [AI computer chips](https://interestingengineering.com/ai-computer-chip-smells-danger-could-replace-sniffer-dogs) will be in use in the near future.

Using supervised learning to detect diseases with similar symptoms may not be an efficient idea. Perhaps in the near future, self-supervised learning models will be trained on detailed human body models such as drones trained in a computer game environment. And these models can be combined with AI computer chips that will be implanted in the human body to monitor.
(For the visual and auditory intuition from childhood:
***References: [link1](https://www.youtube.com/watch?v=7AWfzy7wdv4), [link2](https://www.youtube.com/watch?v=cQUf7xma46o)***)

---

*Suggestions to improve the covid detection accuracy in this study:*

There are publications that apply the SVM method for the output of the network and have been shown to be successful.

A person has the option of not classifying the X-ray image in the learning process if it is not covid and does not know what it is. This situation could be applied to neural network training. Adding a class which doesn't classifies x-ray images but it classifies 1 channel grayscale images such as random lines, circles might help to prevent overfitting and help to generalize the model.

<p>&nbsp;</p>

---
**Abstract**

Aim of this study is designing a deep learning model to detect Covid 19 disease using x-ray lung images from “Kaggle, CoronaHack -Chest X-Ray-Dataset” dataset. This dataset has unlabeled CT images and labeled X-ray images. There are 7 different categories in total, 1 normal and 6 disease with lung inflammation symptoms. Kaggle dataset was highly unbalanced. Therefore, compared to the network architecture,  more study was made on the efficient use of dataset that is detected to have faulty labels. For the neural network, Fully Connected Layer and Dropout, on top of pretrained ResNet18 was used. By using different type of methods on dataset and babysitting the training process; 83% accuracy was achieved on CoronaHack Test dataset. Observing the behaviour of the training, validation and test set results, a methodology was followed. Sample sizes of classes and randomly picked images was changed during the training according to metrics. When the model found a local minimum that could not be generalized for covid19 detection, the learning rate was increased up to 0.1 to find another local optima. When metrics did not improve due to the learning rate, the images were resampled and the sample sizes of each class were changed in favor of incorrectly predicted labels. To check the applicability of sampling method, class weights of the model was changed in favor of covid class. With these settings, model was able to predict covid19 images succesfully. Therefore sampling method was applied for practising and testing the understanding about deep learning methods. On two different “covid/normal” test datasets 91% and 88% accuracy was achieved. About 90% portion of the wrong predictions belong to a "viral or bacterial inflammation" class while ground truth label was covid19. Number of False Negative is 1 out of 140. Number of False Positive is 2 out of 140. 21 out of 140 Covid19 labeled data predicted as Viral&Bacterial inflammation. Datasets were small and noisy. Therefore model can not be generalized in its current form.

(The motivation carried in this project was being able to merge and apply the knowledge that has been learnt during the Covid19 pandemic o_o )

**1 Introduction**

Covid 19 is classified as a viral infection. In symptomatic and severe cases, it results with an inflammation in the lungs. Inflammation is a immune response of a mamal body in order to protect the damaged body area. This inflamation can be observed as fluid increasement. Covid 19-induced increased fluid can be detected on chest X-ray images as an white foggy area on lungs. Human eye can distinguish lung and other body part on X-ray image in normal cases. During inflamation dark colored lung area can be partly seen. As the fluid increase, white foggy area on x-ray image increases. This white cloud is a common symptom which is observed in other lung diseases as well. Therefore, it was assumed that using inflammation only as a distinguishing feature for covid and classifying all diseases separately would not significantly increase the success rate of Covid19 detection task. With this non evidence-based reasoning and assumption, the classification was divided into 3 different groups as Normal, Viral or Bacterial Inflammation and Covid19 Inflamation. This assumption has not been thoroughly tested due to lack of computation power. For computation Google Colab and free provided Google Colab Nvidia T4/16Gb GPU were used.

**2 Dataset**

Kaggle Coronahack Chest X Ray dataset disease categories can be seen on Table 1.  Dataset after dividing into 3 different group can be seen on Table 2. "Train" and "Test" seperation is belong to Kaggle dataset. However, there was no covid image in the test file. For this reason 3 randomly selected Covid images was moved from training to test folder. Validation set was created from training set with the same distribution.


CoronaHack **Training** Datas *(Table2)*

|**Category**|**Symptom**|**Class Number**|**Quantity**|
|:------------------|:----------:|:---------:|:-------:|
|Normal                   |Normal      |0          |1342 |
|Others                   |Inflamation |(1,2,4,5,6)|3886 |
|Covid19                  |Inflamation |3          |55   |

CoronaHack **Test** Datas *(Table2)*

|**Category**|**Symptom**|**Class Number**|**Quantity**|
|:------------------|:----------:|:---------:|:-------:|
|Normal                   |Normal      |0          |234 |
|Others                   |Inflamation |(1,2,4,5,6)|390 |
|Covid19                  |Inflamation |3          |3   |  


Network training ***without changing class weights or changing sample sizes*** does not result in a situation where all Covid 19 images are correctly predicted with this dataset. Therefore the Result Model babysitted. 


**2.1 Custom Dataset and Transformations**

For kaggle dataset, csv file has been modified for the custom data loader defined for the pytorch dataloader.

**2.2 Color channels**

X-ray images in the Kaggle dataset have Gray, RGB and RGB-A channels.  For the backbone of the model, Pretrained ResNet18 was used. Input of ResNet18 has 3 channel, 224x224 Input size. Input shape is 3x224x224. All the image channels listed. And result showed that majority of the Covid19 files have **RGB** channels and almost all of the Normal datas have **Gray** channel.

Learning model could learn that RGB images are belong to Covid19 class and Gray images are  belong to Normal class. For this reason and to prevent biases caused by channel dimensions and colors, all non-gray images converted into gray images. For this task “convert” function from “Pillow” library is used (eq.(1)). After this process, to be able to train our ResNet based model, all images converted to RGB with “convert“ function from “Pillow” library (eq.(2)).

|function|Eq|
|--------------------|--:|
|Image.convert(‘L’)  |(1)|
|Image.convert(‘RGB’)|(2)|

**2.3 Horizontal flip**

To be able to use images as a ResNet input; all images resized to 224 pixels with pytorch “Resize” transformation and longer axis has cropped with “CenterCrop” transformation. In order to prevent the biases caused by image position the horizantal flip transform was applied to the image as applied in the ResNet article with the probability of 0.5. 



**2.4 Data normalization parameters**

Images in datasets have different types of attributes such as color and contrast. In order to prevent these differentiation between images, after converting all the images to Grayscale normalization applied to the transformed images. Mean (mu) and Standart Deviation (std) were calculated by using training images of Kaggle, Chest X-ray dataset (eq.(3), eq.(4)).

|Mean and Standart Deviation|Eq|
|:--------------------:|--:|
|mean = 0.570406436920166  |(3)|
|standart deviation = 0.177922099828720)|(4)|


If dataloader use PIL images, Pytorch automatically scales the pixel values between 0 and 1. 
Model trained with “PIL.convert()” channel transformations. 
When test set was transformed with torch GrayScale and RGB channel transformations, test accuracy dropped 
to 32 percent from 88 percent for [Test3 dataset](https://www.kaggle.com/nabeelsajid917/covid-19-x-ray-10000-images?select=dataset). 
This can be interpreted as two different libraries use different methods for channel transformations.  



**3 Deciding on model** 

**3.1 Considering normalization parameters**

Standart deviation of kaggle dataset is smaller than the Imagenet standart deviation[ref]. This could be interpreted as Imagenet that has 12million different images compare to same type of kaggle dataset lung X-ray images have bigger standart deviation. Deep neural network models in this project were able to learn functions with only random initialization at first epoch.  And they overfit easily due to their complexity. Large training datasets can prevent this from happening. Therefore pretrained deep learning models taken into consideration due to small data size of Coronahack dataset. Two types of models are considered for the model. Two types of options was picked. One of them was residual networks and the other was networks that has bottleneck layers like inception network.

**3.2 Eliminating the networks with bottleneck layers**

According to Inception(v3) article [[ref](https://arxiv.org/abs/1512.00567)], in General Design Principles section, “Avoid representational bottlenecks, especially early in the network[[ref](https://arxiv.org/pdf/1512.00567.pdf)]” and “One should avoid bottlenecks with extreme compression[[ref](https://arxiv.org/pdf/1512.00567.pdf)]” These warnings were interpreted for the covid19 classification tasks as; CNN’s can learn necessary features for the classification task but bottleneck layers will add up all the learned features and information will be transformed. With small dataset unlike imagenet, network may not learn new meaningfull features from bottleneck layers. Same intuition carried for the pooling layers which will be explained in the next section. In addition, possibility of vanishing gradients and exploding gradients was a major drawback. For these reasons, residual network became a strong option against inception network.


**3.3 Residual Networks**

**3.3.1 Why “Residual Networks” ?**

The Kaggle dataset has chest X-ray images that the untrained eye can tell they look alike. They may look same from a distance. These X-ray images have small standart deviation.  Transformed images are Gray. Therefore important features to separate Covid and Normal images might have the similar values after few layers depends on the convolution sizes and pooling layers. Lung inflammations look like an white cloud on top of the lungs (edit! This assumption wasn't tested by scientific methods of research). And after few convolution layers electrots and cables may also look like an inflammation to a network. With these reasonings, it was assumed that Residual Networks can operate so that they do not lose the important distinctive features.


**3.3.2 Residual networks and transfer learning**

The residual layers of ResNet will prevent the model from exploding and vanishing gradient results. 
Activation of previous layers will be continually added to forward layers so that useful features can be learned without losing much input information. Having a pretrained Residual Network trained with Imagenet hypothetically will lower the chance of overfitting compare to untrained version. Another strong motivation for using transfer learning is Andrew Ng’s quote from Deep Learning Specialization Course on Coursera. *“In all the different disciplines, in all the different applications of deep learning, I think that computer vision is one where transfer learning is something you should almost always do, unless, you have an exceptionally large dataset to train everything else from scratch, yourself* [[ref](https://youtu.be/FQM13HkEfBk?t=496)]".

**3.3.3 The result of Untrained ResNet Model training**

Training of the Untrained ResNet models resulted with overfitting on first epoch. Model found a local minima and accuracy didn’t improve. Kaggle test set accuracy was very low compare to the validation set accuracy. Resnet models are complex enough to initialize with local minima for the Kaggle, CoronaHack Chest X-ray "train" data set.
