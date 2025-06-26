# MNIST Digit Identifier Neural Netowork
## A fully functional python neural network built using TensorFlow/Keras that learns how to identify handwritten digits from the MNIST database.
This project includes building and training a neural network for digit classification using TensorFlow and Karas, along with visualizing the data using matplotlib. This project:
- Loads the MNIST database of 28x28 images
- Builds a feedforward (one direction) neural network with 784 input neurons, 128 hidden neurons (all in one hidden layerr), and 10 output neurons
- Uses softmax activation in the output layer for multi-class classification (digits 0–9)
- Trains the model on 10,000 images every epoch for 5 epochs
- Evaluates accuracy
- Prints the training as its happening real time every epoch
- Plots a bar chart showing confidence for each digit
- Shows a prediction for a random sample of each digit (1-9) and shows confidence

## 🕓 Version history
### 1.0.0
- Terminal based
- Runs for 5 epochs
- Prints the loss and accuracy every epoch
- Outputs it's final accuracy on the test set
- Outputs a prediction for digits 1-9 in a 3x3 grid with a percentage of confidence
- Plots the most confident and least confident digits in a bar graph

### 1.0.1 
- Plots the confidence in a bar graph for digits 0-9 in the test set overall rather than plotting the confidence for 1 sample of each digit

## 💻 How to run 
### Prerequisites for download
- Python 3.10 or 3.11 installed
- Git installed
### Instructions 
1. Clone the project
   - Open command prompt/terminal and type "git clone https://github.com/Supposeduck30/mnist-digit-identifier-neural-network.git"

2. Now move into it
   - cd mnist-digit-identifier-neural-network

3. Create the virtual environment
   - python -m venv venv

4. Activate the virtual environment
   - On Windows, the command is "venv\Scripts\activate"
   - On macOS or Linux, the command is "source venv/bin/activate"

5. Install the required packages
   - pip install tensorflow matplotlib numpy (It may take a while to install everything)

6. Run the project
   - python mnist_neural_network.py

### IDE method 
1. Open your IDE and create a new project/folder

2. Create the virtual environment
   - Open the IDE's terminal and put in "python -m venv venv"

3. Activate the virtual environment
   - On Windows, the command is "venv\Scripts\activate"
   - On macOS or Linux, the command is "source venv/bin/activate"

4. Install the required packages 
   - "pip install tensorflow matplotlib numpy"
  
5. Paste the code into your "main.py" (or whatever you named it)

6. Run it

## 🔧 How to tweak the project for your own uses 
1. Fork the repository

2. Clone the fork

3. Make your changes to the code

4. Commit and push your changes to the fork

5. OPTIONAL - Create a pull request if you want the main repository to change the code with what you changed

## 🧠 How it works 
#### 1. Inputs go in
   - Each image is a grayscale grid with 28x28 pixels
   - Pixel values are scaled between 0 and 1

#### 2. The neurons do math 
   - The first layer of input neurons flatten the 28x28 image into 784 numbers
   - The second layer has 128 neurons using ReLU activation (ReLU activation essentially cuts off negative signals and passes through positive ones)
     - Each neuron takes all 784 inputs, does math, and keeps only positive results
     - ReLU activation article - https://builtin.com/machine-learning/relu-activation-function
   - The third layer of 10 neurons (Digits 0-10) uses Softmax (Adds up probabilities to 1)
     - Outputs 10 probabilities that add up to 1

#### 3. It guesses an answer 
   - The number with the highest confidence is the answer

#### 4. It checks the answer
   - The model compares its guess to the real number

#### 5. It then changes its math to be more accurate

#### 6. It does this loop 5 times 

#### 7. It tests it's skills on the test set
   - 10,000 images it hasn't seen before
   - Prints out it's accuracy on the test set

#### 8. It shows predictions with confidence
   - It picks one random image for digits 1-9
   - It guesses each one and shows the confidence with it

#### 9. The program outputs a bar graph for how confident the model was for each digit 

## Screenshots
#### ![image](https://github.com/user-attachments/assets/a1b83671-7019-488a-ad48-bafa4438cf46)
#### ![image](https://github.com/user-attachments/assets/dd3ac692-8913-40a5-b308-c888910e858a)
### ![image](https://github.com/user-attachments/assets/cc7e7be2-ff1b-4605-af78-a5b1d08f429c)


## ⚠️ Known issues 
- It trains on digits 0-9, but ouputs digits 1-9 to make it a 3x3 grid (which looks better)
