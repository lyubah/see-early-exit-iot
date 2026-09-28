"""
author: Dina Hussein and Lubah Nelson 
sources: https://github.com/biggsbenjamin/earlyexitnet/blob/main/src/earlyexitnet/models/Branchynet.py
and 
    
    RFr
Branchy Neural Network with Early Exit Blocks Documentation:

EarlyExitBlock:
    A module representing an early exit block in the branchy neural network.
    
    Parameters:
        input_features (int): The number of input features to the early exit block.
        num_classes (int): The number of output classes for classification.
        
    Methods:
        __init__(input_features, num_classes):
            Initializes the layers of the early exit block including adaptive average pooling,
            flattening, linear layers, ReLU activation, dropout, and final linear layer.
        
        forward(x):
            Performs a forward pass through the layers of the exit block.
            Computes output probabilities, confidence scores, and checks for early exit conditions.

CNN1D_extended_EENN:
    A neural network model with early exit blocks inserted at specific points.
    
    Parameters:
        in_channels (int): The number of input channels to the model.
        out_classes (int): The number of output classes for classification.
        input_sequence_length (int): The length of the input sequence.
        threashold (float): The threshold value for early exit condition.
        
    Methods:
        __init__(in_channels, out_classes, input_sequence_length, T_n):
            Initializes the model and defines early exit blocks at specified points.
        
        forward(x):
            Defines the forward pass of the model.
            Applies convolutional layers, activation functions, and max-pooling layers.
            Checks for early exit conditions using early exit blocks.
        
train_model:
    Trains the model with early exit blocks inserted.
    
    Parameters:
        model (nn.Module): The neural network model with early exit blocks.
        train_loader (DataLoader): The data loader for training data.
        loss (nn.Module): The loss function for calculating training loss.
        num_epochs (int): The number of epochs for training.
        
    Returns:
        None
        
    Notes:
       This is not yet generalizable to different models! 
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from NN_functions import CNN1D_extended
import pickle 
import math
import pandas
import numpy as np
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device('cpu')


logits=0
entropy =0
is_confident= 0

class LateDataInput(nn.Module):
    def __init__(self, in_channels,output_channels, maxpool_data):
        super(LateDataInput, self).__init__()
        self.conv = nn.Conv1d(in_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_data[0], dilation = maxpool_data[1],
                                    padding=maxpool_data[2])
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class EarlyExitBlock(nn.Module):
    
    def __init__(self, input_sequence_length, num_classes, threshold):
        super(EarlyExitBlock, self).__init__()
        self.threshold = threshold
        self.fc1_input_size = input_sequence_length  # Adjust based on your architecture
        # Define the layers used in the classifier
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()  # ReLU activation layer
        
        
    def early_exit_classifier(self, x):
        #print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the features
        #print(x.shape)
        x = self.fc1(x)  # Apply the first linear transformation FIRST POINT OF FAILURE
        x = self.relu(x)  # Apply ReLU activation function
        x = self.fc2(x)  # Apply the second linear transformation
        return x 

    def forward(self, x): 
        logits = self.early_exit_classifier(x)  # The output of the classifier
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-5), dim=1)  # Calculate entropy for confidence measure
        is_confident = self.get_confidence_measure(entropy)  # Check if the output is confident
        #print(is_confident)
        return {
            'logits': logits,
            'entropy': entropy,
            'is_confident': is_confident
        }

    def get_confidence_measure(self, entropy):
        # Lower entropy indicates higher confidence
        is_confident = (entropy <= self.threshold)  # True if entropy is lower than threshold (indicating confidence)
        return is_confident


class CNN1D_extended_EENN(nn.Module):
    
    #TO DO: Initiate model correctly, input to correct parameters, init () may not need it, and find a place to load and call the code
    # Decide if you need nn.MOdeil(for EE blocks or not)
    
    """
    A neural network model with early exit blocks inserted at specific points.
    """

    def __init__(self, in_channels, out_classes, input_sequence_length, thresholds, num_exits, is_training, n_data, exit_placement):        
        """
        Initializes the model and defines early exit blocks at specified points.
        
        Parameters:
            in_channels (int): The number of input channels to the model.
            out_classes (int): The number of output classes for classification.
            input_sequence_length (int): The length of the input sequence.
            thresholds (float): array with the threshold value for early exit condition.
            num_exits (int): number of exits to be included
            is_traing(Bool): A boolean that will decide which forward method to use
            
        """
               
        super(CNN1D_extended_EENN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.is_training = is_training


        self.num_exits = num_exits + 1
        
        
        self.exit_placements = exit_placement #[1,3]
        # parameters for output_dim:
        self.n_data = n_data
        
        maxpool_size = (self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding, self.maxpool.dilation)
        conv_layers = [(3, 8, 3, 1), (8, 16, 3, 1), (16, 32, 3, 1), (32, 64, 3, 1), (64, 128, 3, 1)]


        #new_length = math.floor((n_data * self.new_data_percentage) / 100)
        #n_data = new_length
        self.EE_dims = calculate_output_dim(n_data, conv_layers, maxpool_size)
        
        self.fc1_input_size = self.EE_dims[-1]  #75 for shoaib, 192 for pamap
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, out_classes)


        # Initate mod list for ee
        # Check if the number of thresholds matches the number of exits
      
        # Initialize a ModuleList to hold your early exit blocks
        self.early_exits = nn.ModuleList()

        # # Define early exits
        # for i, t_n in enumerate(thresholds): 
        #     exit_block = EarlyExitBlock(input_sequence_length = self.EE_dims[i],
        #                                 num_classes = out_classes, threshold = t_n) 
        # Define early exits at specified layers using dimensions from EE_dims
            # self.early_exits.append(exit_block)
        th = 0 
        
        for idx in self.exit_placements:
            
            if idx == 5:
                # Placement 5 is the final classifier, which already exists; no EarlyExitBlock is needed.
                continue
            elif 0 < idx <= 4:
                # Early exits for idx 1 to 4
                exit_in_dimension = self.EE_dims[idx-1]
                threshold = thresholds[th]
                
                exit_block = EarlyExitBlock(exit_in_dimension, out_classes, threshold)
                self.early_exits.append(exit_block)
                th += 1

        # out_conv_channel = self.output_channelsL[th]
        # max_kernel_size = self.kernel_sizeL[th]
        # late_data_input = LateDataInput(out_conv_channel, max_kernel_size)
        # self.late_input.append(late_data_input)


    def forward(self, x, is_training):
        """
        Defines the forward pass of the model.
        
        Parameters:
            x (Tensor): The input tensor to the model.
            training (bool): Flag indicating if the model is in training mode.
            
        Returns:
            Dictionary containing output, exit point, and confidence if not training.
            Model's final output if training.
        """
        # x has shape (batch_size, in_channels, sequence_length)

        if is_training:
            # Forward pass for training to byoass return inexit blocks
            return self.forward_train(x)
        else:
            # Forward pass for inference, using early exits
            return self.forward_inference(x)


    def forward_train(self, x):
        
        
        """
        NOTE ON FORWARD TRAIN: 
            We may not want to train every exiy together! This is for training every exit together
        
        Parameters:
            x (Tensor): The input tensor to the model.
            
        Returns:
            List[Tensor]: The outputs from all exits and the final classifier of the model.
        """

        exit_id = 0
        exits_outputs = []
        
        layer_id = 1
        
        
        # Process input through each layer and early exit
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        # Collect output from the first exit for training
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x1)['logits'])
            exit_id +=1
        
        layer_id = layer_id + 1
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x2)['logits'])
            exit_id +=1


        layer_id = layer_id + 1
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x3)['logits'])
            exit_id +=1
        
        layer_id = layer_id + 1

        x4 = self.conv4(x3)
        x4 = self.relu(x4)
        x4 = self.maxpool(x4)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x4)['logits'])
            exit_id +=1
            
        
        x5 = self.conv5(x4)
        x5 = self.relu(x5)
        x5 = self.maxpool(x5)   
        
        x5 = x5.view(x.size(0), -1)


        # Final classifier output
        final_output = self.fc1(x5)
        final_output = self.relu(final_output)
        final_output = self.fc2(final_output)
    
        # Collect final classifier output for training
        exits_outputs.append(final_output)
    
        return exits_outputs
    
#  For merging: just added updated forward and forward inference
    def forward_inference(self, x):
        """
         Collects data instead of Early exit inference.
     
         Parameters:
             x (Tensor): The input tensor to the model.
     
         Returns:
             List of dictionaries containing outputs, exit points, and confidences for all processed exits.
         """
        results = []  # List to store results from each exit point
        idx = 0  # Index to track which exit we are at
    
        # Define the sequence of operations for each layer
        operations = [
            (self.conv1, self.relu, self.maxpool, 1),
            (self.conv2, self.relu, self.maxpool, 2),
            (self.conv3, self.relu, self.maxpool, 3),
            (self.conv4, self.relu, self.maxpool, 4),
            (self.conv5, self.relu, self.maxpool, 5)
        ]
    
        # Iterate through each layer
        for conv, activation, pool, layer_num in operations:
            x = conv(x)  # Apply convolution
            x = activation(x)  # Apply activation function
            x = pool(x)  # Apply pooling
    
            # Check if this layer is an early exit point
            # The last layer is the final classifier, not an early exit, so it is skipped here
            if layer_num in self.exit_placements and layer_num != 5:
                # Process the early exit
                exit_output = self.early_exits[idx](x)
                
                _, exit_label = torch.max(exit_output['logits'].data, 1)
                
                batch_size = np.size(exit_label.detach().cpu().numpy())
                # Record the exit's output
                results.append({
                    'output': exit_label.detach().cpu().numpy(),  # Convert tensor to NumPy array
                    'exit_point': np.ones(batch_size, dtype=int)*layer_num,
                    'early_exit': np.ones(batch_size, dtype=int)*1,
                    'threshold_value': np.ones(batch_size)*self.early_exits[idx].threshold,  
                    'confidence': exit_output['entropy'].detach().cpu().numpy(),
                    'is_confident': exit_output['is_confident'].cpu().numpy()
                })
                idx += 1  # Move to the next exit for the next iteration
            else:
                # No exit at this layer: record a -1 placeholder so results stay indexed by layer
                batch_size , _, _= x.cpu().detach().numpy().shape
                results.append({
                    'output': -1*np.ones(batch_size, dtype=int),   
                    'exit_point': np.ones(batch_size, dtype=int)*-1,
                    'early_exit': np.ones(batch_size, dtype=int)*0,
                    'threshold_value': -1*np.ones(batch_size, dtype=int),   
                    'confidence': np.ones(batch_size, dtype=int)*-1,
                    'is_confident': np.ones(batch_size, dtype=int)*-1
                    })

        # remove last element from list
        # drop the placeholder for the last layer; the final classifier result is appended below
        results.pop()
        # If no early exit was taken, process through the rest of the network
        # if not results[-1]['is_confident']:  # Check if the last processed exit was not confident
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        final_output = self.fc2(self.relu(self.fc1(x)))  # Final classifier output
        _, final_label = torch.max(final_output.data, 1)
        
        batch_size = np.size(final_label.detach().cpu().numpy())
        # Record the final output
        results.append({
            'output': final_label.detach().cpu().numpy(),  # Convert tensor to NumPy array
            'exit_point': np.ones(batch_size, dtype=int)*layer_num,# final exit is recorded at the last layer index
            'early_exit': np.ones(batch_size, dtype=int)*0,
            'confidence': np.ones(batch_size, dtype=int)*-1,  # Final exit does not have an associated confidence
            'threshold_value': np.ones(batch_size)*-1,
            'is_confident': np.ones(batch_size)*True  # Final output is always 'confident' as it's the last resort
        })
    
        return results


    def inference(self, train_loader, test_loader):
        self.eval()  # Switches the model to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        results = []
        train_test_track = [] # 1 for train, 0 for test
        
        
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                outputs = self.forward(inputs, self.is_training) 
                results.append(outputs)
                labels_array = labels.detach().cpu().numpy()
                labels_size = labels_array.size
                curr_label = dict()
                curr_label["label"] = labels_array
                curr_label["train_test"] = np.ones(labels_size, dtype=int)
                train_test_track.append(curr_label)
        
    
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = self.forward(inputs, self.is_training) 
                results.append(outputs)
                labels_array = labels.detach().cpu().numpy()
                labels_size = labels_array.size
                curr_label = dict()
                curr_label["label"] = labels_array
                curr_label["train_test"] = np.zeros(labels_size, dtype=int)
                train_test_track.append(curr_label)
                
        return results, train_test_track


    # CHANGE MADE HERE: Commented out training method that printed, and simply added  it with return a results
    def train_model(self, train_loader, test_loader, loss_weights, num_epochs):
           """
           Trains the model with early exit blocks inserted and returns training and testing data.
           
           """
           # Define optimizer and loss function
           optimizer = Adam(self.parameters(), lr=0.001)
           criterion = CrossEntropyLoss()

           # Move model to the device
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           self.to(device)

           # Initialize structure to hold training and testing results
           results = []

           for epoch in range(num_epochs):
               #self.train()  # Set model to training mode
               total_loss = 0.0
               total = 0
               correct = np.zeros((self.num_exits, 1))

               for i, (batch_data, batch_labels) in enumerate(train_loader):
                   batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                   optimizer.zero_grad()
                   outputs = self(batch_data, True)

                   batch_labels = batch_labels.type(torch.long)   

                   exit_losses = [criterion(output, batch_labels) for output in outputs]
                   if loss_weights:
                       exit_losses = [weight * loss for weight, loss in zip(loss_weights, exit_losses)]
                   
                   loss = sum(exit_losses)
                   loss.backward()
                   optimizer.step()

                   total_loss += loss.item()
                   total += batch_labels.size(0)

                   for idx, o in enumerate(outputs):
                       _, predicted = torch.max(o.data, 1)
                       correct[idx] += (predicted == batch_labels).sum().item()

               train_accuracy = 100 * correct / total
               #print(epoch)

               # Testing the model with test set and getting the accuracy at all exits
               #self.eval()  # Set model to evaluation mode
               correct_test = np.zeros((self.num_exits, 1))
               total_test = 0

               with torch.no_grad():
                   for i, (batch_data, batch_labels) in enumerate(test_loader):
                       batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                       outputs = self(batch_data, True)

                       batch_labels = batch_labels.type(torch.long)
                       total_test += batch_labels.size(0)

                       for idx, o in enumerate(outputs):
                           _, predicted = torch.max(o.data, 1)
                           correct_test[idx] += (predicted == batch_labels).sum().item()

               test_accuracy = 100 * correct_test / total_test

               # Append epoch results
               results.append({
                   'epoch': epoch + 1,
                   'train_accuracy': train_accuracy.flatten().tolist(),
                   'test_accuracy': test_accuracy.flatten().tolist()
               })
        
           return results
        
        
# Initialize the model with exits 
class CNN1D_extended_EENN_partialsampling(nn.Module):
    
    #TO DO: Initiate model correctly, input to correct parameters, init () may not need it, and find a place to load and call the code
    # Decide if you need nn.MOdeil(for EE blocks or not)
    
    """
    A neural network model with early exit blocks inserted at specific points.
    """
    
    def __init__(self, in_channels, out_classes, input_sequence_length, thresholds, num_exits, is_training, n_data, exit_placement, new_data_percentage):        
        """
        Initializes the model and defines early exit blocks at specified points.
        
        Parameters:
            in_channels (int): The number of input channels to the model.
            out_classes (int): The number of output classes for classification.
            input_sequence_length (int): The length of the input sequence.
            thresholds (float): array with the threshold value for early exit condition.
            num_exits (int): number of exits to be included
            is_traing(Bool): A boolean that will decide which forward method to use
            
        """
               
        super(CNN1D_extended_EENN_partialsampling, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.is_training = is_training


        self.num_exits = num_exits + 1
        
        
        self.exit_placements = exit_placement #[1,3]
        # parameters for output_dim:
        self.n_data = n_data
        
        maxpool_size = (self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding, self.maxpool.dilation)
        conv_layers = [(3, 8, 3, 1), (8, 16, 3, 1), (16, 32, 3, 1), (32, 64, 3, 1), (64, 128, 3, 1)]


        self.new_data_percentage = new_data_percentage
        
        data_used_partial = np.sum(self.new_data_percentage)
        
        self.cumulative_data_percentage = self.new_data_percentage + [100 - data_used_partial]
        
        #new_length = math.floor((n_data * self.new_data_percentage) / 100)
        #n_data = new_length
        #self.EE_dims = calculate_output_dim(n_data, conv_layers, maxpool_size)
        self.output_channelsL, self.kernel_sizeL, self.EE_dims, self.data_ids = calculate_late_in_dim(n_data, conv_layers, maxpool_size, exit_placement, new_data_percentage)
        if self.output_channelsL == -1:
            return
        self.fc1_input_size = self.EE_dims[-1]  #75 for shoaib, 192 for pamap
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, out_classes)
        
        
        self.late_input = nn.ModuleList()


        # Initate mod list for ee
        # Check if the number of thresholds matches the number of exits
      
        # Initialize a ModuleList to hold your early exit blocks
        self.early_exits = nn.ModuleList()

        # # Define early exits
        # for i, t_n in enumerate(thresholds): 
        #     exit_block = EarlyExitBlock(input_sequence_length = self.EE_dims[i],
        #                                 num_classes = out_classes, threshold = t_n) 
        # Define early exits at specified layers using dimensions from EE_dims
            # self.early_exits.append(exit_block)
        th = 0 
        
        for idx in self.exit_placements:
            if idx == 5:
                continue
            elif 0 < idx <= 4:
                # Early exits for idx 1 to 4
                exit_in_dimension = self.EE_dims[idx-1]
                threshold = thresholds[th]
                
                exit_block = EarlyExitBlock(exit_in_dimension, out_classes, threshold)
                self.early_exits.append(exit_block)
                
                out_conv_channel = self.output_channelsL[th]
                max_kernel_size = self.kernel_sizeL[th]
                late_data_input = LateDataInput(in_channels,out_conv_channel, max_kernel_size)
                self.late_input.append(late_data_input)

                th +=1
        # out_conv_channel = self.output_channelsL[th]
        # max_kernel_size = self.kernel_sizeL[th]
        # late_data_input = LateDataInput(out_conv_channel, max_kernel_size)
        # self.late_input.append(late_data_input)


    def forward(self, x, is_training):
        """
        Defines the forward pass of the model.
        
        Parameters:
            x (Tensor): The input tensor to the model.
            training (bool): Flag indicating if the model is in training mode.
            
        Returns:
            Dictionary containing output, exit point, and confidence if not training.
            Model's final output if training.
        """
        # x has shape (batch_size, in_channels, sequence_length)

        if is_training:
            # Forward pass for training to byoass return inexit blocks
            return self.forward_train(x)
        else:
            # Forward pass for inference, using early exits
            return self.forward_inference(x)


    def forward_train(self, x):
        
        
        """
        NOTE ON FORWARD TRAIN: 
            We may not want to train every exiy together! This is for training every exit together
        
        Parameters:
            x (Tensor): The input tensor to the model.
            
        Returns:
            List[Tensor]: The outputs from all exits and the final classifier of the model.
        """

        exit_id = 0
        exits_outputs = []
        
        layer_id = 1
        
        data_ids = self.data_ids
        if np.all(data_ids == 0):
            return -1
        x_partial = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
        
        # Process input through each layer and early exit
        x1 = self.conv1(x_partial)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        # Collect output from the first exit for training
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x1)['logits'])
            exit_id +=1
            
            
            new_sensor_block = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
            x_new = self.late_input[exit_id-1](new_sensor_block)
            x1 = torch.cat((x1, x_new), dim=2)
        
        layer_id = layer_id + 1
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x2)['logits'])
            exit_id +=1
            
            new_sensor_block = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
            x_new = self.late_input[exit_id-1](new_sensor_block)
            x2 = torch.cat((x2, x_new), dim=2)


        layer_id = layer_id + 1
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x3)['logits'])
            exit_id +=1
            
            new_sensor_block = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
            x_new = self.late_input[exit_id-1](new_sensor_block)
            x3 = torch.cat((x3, x_new), dim=2)
        
        layer_id = layer_id + 1

        x4 = self.conv4(x3)
        x4 = self.relu(x4)
        x4 = self.maxpool(x4)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x4)['logits'])
            exit_id +=1
            
            new_sensor_block = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
            x_new = self.late_input[exit_id-1](new_sensor_block)
            x4 = torch.cat((x4, x_new), dim=2)
            
        
        x5 = self.conv5(x4)
        x5 = self.relu(x5)
        x5 = self.maxpool(x5)   
        
        x5 = x5.view(x.size(0), -1)


        # Final classifier output
        final_output = self.fc1(x5)
        final_output = self.relu(final_output)
        final_output = self.fc2(final_output)
    
        # Collect final classifier output for training
        exits_outputs.append(final_output)

        return exits_outputs


    def forward_inference(self, x):
        """
         Collects data instead of Early exit inference.
     
         Parameters:
             x (Tensor): The input tensor to the model.
     
         Returns:
             List of dictionaries containing outputs, exit points, and confidences for all processed exits.
         """
        All_x = copy.deepcopy(x)
        exit_id = 0
        
        cumulative_data_used = 0
                 
        data_ids = self.data_ids
        x_partial = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
         
        results = []  # List to store results from each exit point
    
        # Define the sequence of operations for each layer
        operations = [
            (self.conv1, self.relu, self.maxpool, 1),
            (self.conv2, self.relu, self.maxpool, 2),
            (self.conv3, self.relu, self.maxpool, 3),
            (self.conv4, self.relu, self.maxpool, 4),
            (self.conv5, self.relu, self.maxpool, 5)
        ]
    
        # Iterate through each layer
        for conv, activation, pool, layer_num in operations:
            if layer_num == 1:
                cumulative_data_used = cumulative_data_used + self.cumulative_data_percentage[exit_id]
                x = x_partial
            x = conv(x)  # Apply convolution
            x = activation(x)  # Apply activation function
            x = pool(x)  # Apply pooling


            # Check if this layer is an early exit point
            if layer_num in self.exit_placements:
                # Process the early exit
                exit_output = self.early_exits[exit_id](x)
                
                _, exit_label = torch.max(exit_output['logits'].data, 1)
                
                batch_size = np.size(exit_label.detach().cpu().numpy())
                # Record the exit's output
                results.append({
                    'output': exit_label.detach().cpu().numpy(),  # Convert tensor to NumPy array
                    'exit_point': np.ones(batch_size, dtype=int)*layer_num,
                    'early_exit': np.ones(batch_size, dtype=int)*1,
                    'threshold_value': np.ones(batch_size)*self.early_exits[exit_id].threshold,  # Inserted threshold value here
                    'confidence': exit_output['entropy'].detach().cpu().numpy(),
                    'is_confident': exit_output['is_confident'].cpu().numpy(),
                    'partial_percentage': np.ones(batch_size)*cumulative_data_used
                })
                exit_id += 1  # Move to the next exit for the next iteration
                
                cumulative_data_used = cumulative_data_used + self.cumulative_data_percentage[exit_id]
                new_sensor_block = All_x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
                x_new = self.late_input[exit_id-1](new_sensor_block)
                x = torch.cat((x, x_new), dim=2)
            else:
                
                batch_size , _, _= x.detach().cpu().numpy().shape
                results.append({
                    'output': -1*np.ones(batch_size, dtype=int),   
                    'exit_point': np.ones(batch_size, dtype=int)*-1,
                    'early_exit': np.ones(batch_size, dtype=int)*0,
                    'threshold_value': -1*np.ones(batch_size, dtype=int),   
                    'confidence': np.ones(batch_size, dtype=int)*-1,
                    'is_confident': np.ones(batch_size, dtype=int)*-1,
                    'partial_percentage': np.ones(batch_size, dtype=int)*-1
                    })

        results.pop()


                # # Break out of the loop if the exit is confident
                # if exit_output['is_confident']:
                #     break
    
        # If no early exit was taken, process through the rest of the network
        # if not results[-1]['is_confident']:  # Check if the last processed exit was not confident
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        final_output = self.fc2(self.relu(self.fc1(x)))  # Final classifier output
        _, final_label = torch.max(final_output.data, 1)
        
        batch_size = np.size(final_label.detach().cpu().numpy())
        # Record the final output
        results.append({
            'output': final_label.detach().cpu().numpy(),  # Convert tensor to NumPy array
            'exit_point': np.ones(batch_size, dtype=int)*layer_num,
            'early_exit': np.ones(batch_size, dtype=int)*0,
            'confidence': np.ones(batch_size, dtype=int)*-1,  # Final exit does not have an associated confidence
            'threshold_value': np.ones(batch_size)*-1,
            'is_confident': np.ones(batch_size)*True, # Final output is always 'confident' as it's the last resort
            'partial_percentage': np.ones(batch_size, dtype=int)*cumulative_data_used

        })
    
        return results

    def inference(self, train_loader, test_loader):
        self.eval()  # Switches the model to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        results = []
        train_test_track = [] # 1 for train, 0 for test
        
        
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                outputs = self.forward(inputs, self.is_training) 
                results.append(outputs)
                labels_array = labels.detach().cpu().numpy()
                labels_size = labels_array.size
                curr_label = dict()
                curr_label["label"] = labels_array
                curr_label["train_test"] = np.ones(labels_size, dtype=int)
                train_test_track.append(curr_label)
        
    
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = self.forward(inputs, self.is_training) 
                results.append(outputs)
                labels_array = labels.detach().cpu().numpy()
                labels_size = labels_array.size
                curr_label = dict()
                curr_label["label"] = labels_array
                curr_label["train_test"] = np.zeros(labels_size, dtype=int)
                train_test_track.append(curr_label)
                
        return results, train_test_track

   
    def train_model(self, train_loader, test_loader, loss_weights, num_epochs):
      """
      note on this method
            # We could modify the training loop to stop processing an input batch
            # after an exit condition is met. This requires defining and checking exit 
            # conditions during training. This will be challenging though, as it changes 
          # the distribution of data seen by later parts of the model.
          
          
          Trains the model with early exit blocks inserted.
          
          Parameters:
              model (nn.Module): The neural network model with early exit blocks.
              train_loader (DataLoader): The data loader for training data.
              loss (nn.Module): The loss function for calculating training loss.
              num_epochs (int): The number of epochs for training.
              
          Returns:
              None
              
              
      Loss Function: The softmax cross entropy loss function is utilized, defined for a given exit branch
      Softmax Function: The predicted probabilities ŷ are calculated using the softmax function applied to the outputs z from each exit branch 
      Exit Branch Output: Exit brand output z = f_exit ( x ; theta)  is the result of processing the input sample x through the branch, where theta represents the parameters from an entry point to the exit point. 


      """
      
      
      # Define optimizer and loss function
      optimizer = Adam(self.parameters(), lr=0.001)
      criterion = CrossEntropyLoss()
  
      # Move model to the  device
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.to(device)
      results = []
  
      for epoch in range(num_epochs):
          #self.train()
          total_loss = 0.0
          total = 0
          correct = np.zeros((self.num_exits, 1))
          for i, (batch_data, batch_labels) in enumerate(train_loader):
              # Zero the gradients, forward pass, and compute loss
              batch_data = batch_data.to(device)
              batch_labels = batch_labels.to(device)
              optimizer.zero_grad()
              outputs = self(batch_data, True)
              if outputs == -1:
                  return -1
              
              
              batch_labels = batch_labels.type(torch.long) # check if this is needed
              
              exit_losses = [criterion(output, batch_labels) for output in outputs]
              
              if loss_weights: 
                  exit_losses = [weight * loss for weight, loss in zip(loss_weights, exit_losses)]
              
              loss = sum(exit_losses)
              
              loss.backward()
              optimizer.step()
              
              
              total_loss += loss.item()
              
              
              total += batch_labels.size(0)
              
              for idx, o in enumerate(outputs):
                       _, predicted = torch.max(o.data, 1)
                       correct[idx] += (predicted == batch_labels).sum().item()
    
          train_accuracy = 100 * correct / total
          #print("Epoch ", epoch+1, "Train Accuracy = ", np.transpose(train_accuracy))
          
          
          #Testing the model with test set and geting the accuracy at all exits
          #self.eval()
          correct_test = np.zeros((self.num_exits, 1))
          total_test = 0
    
          with torch.no_grad():
            for i, (batch_data, batch_labels) in enumerate(test_loader):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = self(batch_data, True)
    
                batch_labels = batch_labels.type(torch.long)
                total_test += batch_labels.size(0)
    
                for idx, o in enumerate(outputs):
                    _, predicted = torch.max(o.data, 1)
                    correct_test[idx] += (predicted == batch_labels).sum().item()
            
            test_accuracy = 100 * correct_test / total_test
          
              # Append epoch results
            results.append({
                'epoch': epoch + 1,
                'train_accuracy': train_accuracy.flatten().tolist(),
                'test_accuracy': test_accuracy.flatten().tolist() })


      return results


def calculate_output_dim(input_length, conv_layers, maxpool_size):
    """
    Calculates the output dimensions after each conv and max-pool layer.
    Parameters:
    - input_length: The length of the input sequence.
    - conv_layers: A list of tuples (in_channels, out_channels, kernel_size, padding) for each conv layer.
    - maxpool_size: The size of the max-pooling (assumed to be applied after each conv layer).

    Returns:
    - A list of output dimensions after each max-pooling layer.
    """
    output_dims = []
    current_length = input_length
    
    maxpool_kernel = maxpool_size[0]
    maxpool_stride  = maxpool_size[1]
    maxpool_padding = maxpool_size[2]
    maxpool_dilation = maxpool_size[3]

    for (in_channels, out_channels, kernel_size, padding) in conv_layers:
        #  convolution
        current_length = math.floor(((current_length + (2 * padding) - (kernel_size - 1) - 1) / 1) + 1)   
        # them  max-pooling
        current_length = math.floor((current_length + 2*maxpool_padding - maxpool_dilation*(maxpool_kernel - 1) - 1) / maxpool_stride + 1)
        output_dims.append(out_channels * current_length)

    return output_dims

def calculate_late_in_dim(n_data, conv_layers, maxpool_size, exit_placement, data_perc):
   
    # initialize
    output_dims = []
    late_block_output_channels = []
    maxpool_kernel_size_list = []
    
    count = 1 
    
    # calculate input for first layer
    n_data_each_input = np.floor((n_data * np.asarray(data_perc)) / 100)
    
    cumulative_data_used = np.sum(n_data_each_input)
    
    remaining_data = n_data - cumulative_data_used
    
    n_data_each_input = np.append(n_data_each_input, remaining_data)
    
    data_ids = np.zeros((len(exit_placement)+1, 2), dtype=int)
    
    id_begin = 0
    data_used = 0
    for i in range(0, len(n_data_each_input)):
        
        data_used = data_used + n_data_each_input[i]
        
        data_ids[i, 0] = int(id_begin)
        data_ids[i, 1] = int(data_used)
        
        id_begin = data_used


    maxpool_kernel = maxpool_size[0]
    maxpool_stride  = maxpool_size[1]
    maxpool_padding = maxpool_size[2]
    maxpool_dilation = maxpool_size[3]
    # length of current output
    current_length = n_data_each_input[0]
    
    exit_id = 1
    
    for (in_channels, out_channels, kernel_size, padding) in conv_layers:
        current_length = math.floor(((current_length + (2 * padding) - (kernel_size - 1) - 1) / 1) + 1)   
        current_length = math.floor((current_length + 2*maxpool_padding - maxpool_dilation*(maxpool_kernel - 1) - 1) / maxpool_stride + 1)
        output_dims.append(out_channels * current_length)
        if count in exit_placement:
            late_block_output_channels.append(out_channels)
            
            # new input coming in
            new_input_length = n_data_each_input[exit_id]
            
            maxpool_kernel_new = int(np.floor(new_input_length/current_length))
            
            if (maxpool_kernel_new <= 1):
                return -1,-1,-1,0*data_ids
            # check maxpool output
            maxpool_s = maxpool_kernel_new
            maxpool_d = 1
            maxpool_p = 0
        
            new_out_check = math.floor((new_input_length + 2*maxpool_p - maxpool_d*(maxpool_kernel_new - 1) - 1) / maxpool_s + 1)
            
            if(new_out_check == current_length):
                maxpool_kernel_size_list.append((maxpool_kernel_new, maxpool_d, maxpool_p))
            
            #need to pad data
            elif(new_out_check < current_length):
                maxpool_p = (maxpool_kernel_new*current_length - new_input_length + maxpool_d*maxpool_s - \
                    maxpool_d + 1 - maxpool_kernel_new)
                
                maxpool_p = int(np.floor(maxpool_p))
                
                if (maxpool_p > maxpool_kernel_new/2):
                    maxpool_p = int(np.floor(maxpool_kernel_new/2))
                
                new_out_check = math.floor((new_input_length + 2*maxpool_p - maxpool_d*(maxpool_kernel_new - 1) - 1) / maxpool_s + 1)
                
                maxpool_kernel_size_list.append((maxpool_kernel_new, maxpool_d, maxpool_p))
                
            elif(new_out_check > current_length):
                
                maxpool_d = (new_input_length - maxpool_kernel_new*current_length - 1 + maxpool_s)/(maxpool_s - 1)
                
                maxpool_d = int(np.floor(maxpool_d))
                
                new_out_check = math.floor((new_input_length + 2*maxpool_p - maxpool_d*(maxpool_kernel_new - 1) - 1) / maxpool_s + 1)
                
                maxpool_kernel_size_list.append((maxpool_kernel_new, maxpool_d, maxpool_p))


            current_length = current_length*2
            
            exit_id = exit_id + 1
            
        count = count + 1
        

#    #final 
# late_block_output_channels.append(out_channels)
# kernel_size_list.append(input_length//current_length)


    return late_block_output_channels, maxpool_kernel_size_list, output_dims, data_ids

def extract_new_data(data, new_data_percentage):
    batch_size, channels, samples = data.shape
    new_data_samples = int(samples * new_data_percentage / 100)       
    
    step_size = new_data_samples

    new_data = []

    for i in range(0, samples - new_data_samples + 1, step_size):
        new_data.append(data[:, :, i:i + new_data_samples])

    return new_data


        