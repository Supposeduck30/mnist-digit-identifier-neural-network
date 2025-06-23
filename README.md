# MNIST Digit Identifier Neural Netowork
## A fully functional python neural network using TensorFlow/Keras that learns how to identify handwritten digits from the MNIST database.
This project includes building and training a neural network for digit classification using TensorFlow and Karas, along with visualizing the data using matplotlib. This project:
- Loads the MNIST database of 28x28 images
- Builds a feedforward (one direction) neural network with 784 input neurons, 128 hidden neurons (all in one hidden layerr), and 10 output neurons
- Uses softmax activation in the output layer for multi-class classification (digits 0–9)
- Trains the model on 10,000 images every epoch for 5 epochs
- Evaluates accuracy
- Prints the training as its happening real time every epoch
- Plots a bar chart showing confidence for each digit
- Shows a prediction for a random sample of each digit (1-9) and shows confidence

## How to run 
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

## How to tweak the project for your own uses 
1. Fork the repository

2. Clone the fork

3. Make your changes to the code

4. Commit and push your changes to the fork

5. OPTIONAL - Create a pull request if you want the main repository to change the code with what you changed
