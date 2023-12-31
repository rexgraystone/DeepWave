# DeepWave

A convolutional neural network for the classification of Major Depressive Disorder (MDD) using Electroencephalogram (EEG) signals.

## Dataset

The dataset used for to train this model has been merged from two different sources:

1. [Source 1](https://figshare.com/articles/dataset/EEG_Data_New/4244171)
2. [Source 2](https://figshare.com/articles/dataset/EEG-based_Diagnosis_and_Treatment_Outcome_Prediction_for_Major_Depressive_Disorder/3385168)

<table align='center'>
	<tr>
		<td align="center">
			<img src="Images/Healthy_plot.png" alt="Healthy patient sample"/>
			<p>Figure 1: EEG Waves sample of a Healthy patient</p>
		</td>
		<td align="center">
			<img src="Images/MDD_plot.png" alt="MDD patient sample"/>
			<p>Figure 2: EEG Waves sample of a MDD patient</p>
		</td>
	</tr>
</table>

## Model Architecture

<p align='center'>
    <img src="Images/DeepWave_architecture.png" alt='DeepWave Model Architecture' width="20%" height="20%">
    <br>
    <p align='center'>Figure 5: DeepWave Model Architecture</p>
</p>

## Results

DeepWave achieved a training accuracy of 96.62%, training loss of 8.70%, validation accuracy of 87.05%, validation loss of 60.24%. The accuracy can be further improved by training the model for more epochs and modifying the model architecture.

<table>
  <tr>
    <td align="center">
      <img src="Images/DeepWave_accuracy.png" alt="DeepWave Accuracy Plot" />
      <p>Figure 6: DeepWave Accuracy Plot</p>
    </td>
    <td align="center">
      <img src="Images/DeepWave_loss.png" alt="DeepWave Loss Plot" />
      <p>Figure 7: DeepWave Loss Plot</p>
    </td>
  </tr>
</table>