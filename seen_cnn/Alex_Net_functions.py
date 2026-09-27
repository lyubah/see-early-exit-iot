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
# from torchvision import models
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device('cpu')

logits=0
entropy =0
is_confident= 0
    

class EarlyExitBlock(nn.Module):
    
    def __init__(self, input_sequence_length, num_classes, threshold):
        super(EarlyExitBlock, self).__init__()
        self.threshold = threshold
        self.fc1_input_size = input_sequence_length  # Adjust based on your architecture
        # Define the layers used in the classifier
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()  # ReLU activation layer
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc1_input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        
    def early_exit_classifier(self, x):
        
        #print(x.shape)
        # x = x.view(x.size(0), -1)  # Flatten the features
        # #print(x.shape)
        # x = self.fc1(x)  # Apply the first linear transformation FIRST POINT OF FAILURE
        # x = self.relu(x)  # Apply ReLU activation function
        # x = self.fc2(x)  # Apply the second linear transformation
        # return x 
       
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.classifier(x)  # Apply the classifier layers
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


# Old version of AlexNet
class AlexNetEENN(nn.Module):
    
    # Initialize the AlexNetEENN class, inheriting from nn.Module.
    def __init__(self, in_channels, out_classes, input_sequence_length, thresholds, num_exits, is_training, n_data, exit_placement):
        
        super(AlexNetEENN, self).__init__()  # Call to the constructor of the superclass (nn.Module).

       
        # Load a pretrained AlexNet model from torchvision's model zoo.
        # Because we don't want the 2d version, Im recreating the stucture of AlexNet, but with 1d parameters
        # alexnet = models.alexnet(pretrained=True)
        
        
        self.is_training = is_training
        self.num_exits = num_exits + 1
        self.exit_placements = exit_placement #[1,3]
        self.n_data = n_data
        self.thresholds = thresholds 
        self.early_exits = nn.ModuleList()
        self.out_classes = out_classes
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)


        # Extract the feature layers (convolutional layers) from AlexNet.
        self.layer1 = nn.Sequential(
            nn.Conv1d( in_channels , 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            self.maxpool
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            self.maxpool
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            self.maxpool
            # nn.AdaptiveAvgPool1d(6),

        )
        
        # First conv layer (in_channels, out_channels, kernel_size, stride, padding) default stride is 1
        conv_layers = [
        (in_channels, 64, 11, 4, 2),  
        (64, 192,5, 1,2),  
        (192, 384,3,1, 1), 
        (384, 256,3, 1,1),
        (256, 256,3, 1,1)
        ]

        
        maxpool_layers = [1, 1, 0, 0, 1]
        maxpool_size = (self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding, self.maxpool.dilation)
        
        self.EE_dims = calculate_output_dim(n_data, conv_layers, maxpool_size, maxpool_layers)
        feature_length =  self.EE_dims[-1]         


        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear( feature_length , 4096),  #possible ero
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_classes)  
        )


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
                th += 1
        # print(exit_block)


    def forward(self, x,  is_training):
        # The main forward pass which delegates to either training or inference specific methods.
        if self.is_training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)

    def forward_train(self, x):
        # Method to handle the forward pass when training, collecting outputs from all exits.
        exit_id = 0
        exits_outputs = []
        
        layer_id = 1
        out = self.layer1(x)
        if layer_id in self.exit_placements:            
            exits_outputs.append(self.early_exits[exit_id](out)['logits'])
            exit_id +=1
        
        layer_id = layer_id + 1
        out = self.layer2(out)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](out)['logits'])
            exit_id +=1
        
        layer_id = layer_id + 1
        out = self.layer3(out)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](out)['logits'])
            exit_id +=1
        
        layer_id = layer_id + 1
        out = self.layer4(out)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](out)['logits'])
            exit_id +=1
        
        # layer_id = layer_id + 1
        out = self.layer5(out)
        out = out.view(out.size(0), -1) # view(x.size(0), -1)
        
        final_output = self.classifier(out)
        exits_outputs.append(final_output)

        return exits_outputs
    
    def forward_inference(self, x):
        results = []
        idx = 0
    
        operations = [
            (self.layer1, 1),
            (self.layer2, 2),
            (self.layer3, 3),
            (self.layer4, 4),
            (self.layer5, 5)
        ]
    
        for operation, layer_num in operations:
            x = operation(x)
            # print(f"Shape after layer {layer_num}: {x.shape}")  # Debugging statement
            if layer_num in self.exit_placements and layer_num != 5:
                exit_output = self.early_exits[idx](x)
                _, exit_label = torch.max(exit_output['logits'].data, 1)
                batch_size = np.size(exit_label.detach().cpu().numpy())
    
                results.append({
                    'output': exit_label.detach().cpu().numpy(),
                    'exit_point': np.ones(batch_size, dtype=int)*layer_num,
                    'early_exit': np.ones(batch_size, dtype=int)*1,
                    'threshold_value': np.ones(batch_size)*self.early_exits[idx].threshold,
                    'confidence': exit_output['entropy'].detach().cpu().numpy(),
                    'is_confident': exit_output['is_confident'].cpu().numpy()
                })
                idx += 1
            else:
                batch_size, _, _ = x.cpu().detach().numpy().shape
                results.append({
                    'output': -1*np.ones(batch_size, dtype=int),
                    'exit_point': np.ones(batch_size, dtype=int)*-1,
                    'early_exit': np.ones(batch_size, dtype=int)*0,
                    'threshold_value': -1*np.ones(batch_size, dtype=int),
                    'confidence': np.ones(batch_size, dtype=int)*-1,
                    'is_confident': np.ones(batch_size, dtype=int)*-1
                })
        results.pop()
    
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"Shape before classifier: {x.shape}")  # Debugging statement
    
        final_output = self.classifier(x)
        _, final_label = torch.max(final_output.data, 1)
        batch_size = np.size(final_label.detach().cpu().numpy())
    
        results.append({
            'output': final_label.detach().cpu().numpy(),
            'exit_point': np.ones(batch_size, dtype=int)*layer_num,
            'early_exit': np.ones(batch_size, dtype=int)*0,
            'confidence': np.ones(batch_size, dtype=int)*-1,
            'threshold_value': np.ones(batch_size)*-1,
            'is_confident': np.ones(batch_size)*True
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
                    torch.cuda.empty_cache()
                    
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
                # print("Epoch ", epoch+1, "Train Accuracy = ", np.transpose(train_accuracy))
 
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


class AlexNetPartial(nn.Module):
    
    # Initialize the AlexNetEENN class, inheriting from nn.Module.
    def __init__(self, in_channels, out_classes, input_sequence_length, thresholds, num_exits, is_training, n_data, exit_placement, new_data_percentage):  
        
        super(AlexNetPartial, self).__init__() 
        self.is_training = is_training
        self.num_exits = num_exits + 1
        self.exit_placements = exit_placement #[1,3]
        self.n_data = n_data
        self.thresholds = thresholds 
        self.early_exits = nn.ModuleList()
        self.late_input = nn.ModuleList()
        self.out_classes = out_classes
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.new_data_percentage = new_data_percentage
        data_used_partial = np.sum(self.new_data_percentage)
        self.cumulative_data_percentage = self.new_data_percentage + [100 - data_used_partial]
       
        conv_layers = [
        (in_channels, 64, 11, 4, 2),  
        (64, 192,5, 1,2),  
        (192, 384,3,1, 1), 
        (384, 256,3, 1,1),
        (256, 256,3, 1,1)
        ]
        
        maxpool_layers = [1, 1, 0, 0, 1]
        maxpool_size = (self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding, self.maxpool.dilation)


        self.output_channelsL, self.kernel_sizeL, self.EE_dims, self.data_ids =  calculate_late_in_dim(n_data, conv_layers, maxpool_size, exit_placement, self.new_data_percentage, maxpool_layers)
        if self.output_channelsL == -1:
            return 
        
        feature_length =  self.EE_dims[-1]         
        
        # Extract the feature layers (convolutional layers) from AlexNet.
        self.layer1 = nn.Sequential(
            nn.Conv1d( in_channels , 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            self.maxpool
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            self.maxpool
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            self.maxpool
            # nn.AdaptiveAvgPool1d(6),

        )
        
       
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear( feature_length , 4096),  #possible ero
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_classes)  # Adjust num_classes as necessary
        )


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
                th += 1
        
   
    def forward(self, x,  is_training):
        # The main forward pass which delegates to either training or inference specific methods.
        if self.is_training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)

    def forward_train(self, x):
        #START HERE WITH ADAPTING FORWARD
        # Method to handle the forward pass when training, collecting outputs from all exits.
        exit_id = 0
        exits_outputs = []
        layer_id = 1
        data_ids = self.data_ids
        
        
        if np.all(data_ids == 0):
            return -1
        
        x_partial = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]

        x1 = self.layer1(x_partial)
        
        if layer_id in self.exit_placements:            
            exits_outputs.append(self.early_exits[exit_id](x1)['logits'])
            exit_id +=1
            
            new_sensor_block = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
            x_new = self.late_input[exit_id-1](new_sensor_block)
            x1 = torch.cat((x1, x_new), dim=2)
            
            
        layer_id = layer_id + 1
        x2 = self.layer2(x1)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x2)['logits'])
            exit_id +=1
            
            new_sensor_block = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
            x_new = self.late_input[exit_id-1](new_sensor_block)
            x2 = torch.cat((x2, x_new), dim=2)


        layer_id = layer_id + 1
        x3 = self.layer3(x2)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x3)['logits'])
            exit_id +=1
        
            new_sensor_block = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
            x_new = self.late_input[exit_id-1](new_sensor_block)
            x3 = torch.cat((x3, x_new), dim=2)
        
        layer_id = layer_id + 1
        
        x4 = self.layer4(x3)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](x4)['logits'])
            exit_id +=1
            
            new_sensor_block = x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
            x_new = self.late_input[exit_id-1](new_sensor_block)
            x4 = torch.cat((x4, x_new), dim=2)


        x5 = self.layer5(x4)
        out = x5.view(x5.size(0), -1) # view(x.size(0), -1)
        
        final_output = self.classifier(out)
        exits_outputs.append(final_output)

        return exits_outputs
    
    def forward_inference(self, x):
        
        """
        Training error may be coming from the fact that if id == 5 isn't presnet'
        """
        All_x = copy.deepcopy(x)
        exit_id = 0
        cumulative_data_used = 0
        data_ids = self.data_ids
        x_partial = x[:, :, data_ids[exit_id, 0]:data_ids[exit_id, 1]]
        
        results = []
        
        operations = [
            (self.layer1, 1),
            (self.layer2, 2),
            (self.layer3, 3),
            (self.layer4, 4),
            (self.layer5, 5)
        ]
        
        for layer, layer_num in operations:
            if layer_num == 1:
                cumulative_data_used = cumulative_data_used + self.cumulative_data_percentage[exit_id]
                x = x_partial
            x = layer(x)
            
            if layer_num in self.exit_placements:
                exit_output = self.early_exits[exit_id](x)
                _, exit_label = torch.max(exit_output['logits'].data, 1)
                
                batch_size = np.size(exit_label.detach().cpu().numpy())
                results.append({
                    'output': exit_label.detach().cpu().numpy(),  # Convert tensor to NumPy array
                    'exit_point': np.ones(batch_size, dtype=int)*layer_num,
                    'early_exit': np.ones(batch_size, dtype=int)*1,
                    'threshold_value': np.ones(batch_size)*self.early_exits[exit_id].threshold,  # Inserted threshold value here
                    'confidence': exit_output['entropy'].detach().cpu().numpy(),
                    'is_confident': exit_output['is_confident'].cpu().numpy(),
                    'partial_percentage': np.ones(batch_size)*cumulative_data_used
                })
                exit_id += 1
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
       
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        final_output = self.classifier(x)
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
              torch.cuda.empty_cache()
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
          # print("Epoch ", epoch+1, "Train Accuracy = ", np.transpose(train_accuracy))
          
          
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


def extract_new_data(data, new_data_percentage):
    batch_size, channels, samples = data.shape
    new_data_samples = int(samples * new_data_percentage / 100)       
    
    step_size = new_data_samples

    new_data = []

    for i in range(0, samples - new_data_samples + 1, step_size):
        new_data.append(data[:, :, i:i + new_data_samples])

    return new_data


def check_and_terminate(current_length, data_ids):
    """Helper function to check if current_length is zero and return early if needed."""
    if current_length == 0:
        return None, None, None, None


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


def calculate_output_dim(n_data, conv_layers, max_pool_params, max_pool_flags):
    """
    Calculate the output dimensions for each convolutional layer in a neural network,
    optionally followed by a max-pooling operation.

    Parameters:
    - n_data: int, initial length of the data entering the first layer.
    - conv_layers: list of tuples, each containing (in_channels, out_channels, kernel_size, padding, stride)
                   for each convolutional layer.
    - max_pool_params: tuple, parameters for the max-pooling layers (kernel_size, stride, padding, dilation).
    - max_pool_flags: list of int, flags (0 or 1) indicating whether a max-pooling layer follows each convolutional layer.

    Returns:
    - List of output dimensions after each layer.
    """
    output_dims = []
    current_length = n_data


    for idx, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
        # Calculate output dimension after the convolution
        # print(current_length)
        current_length = math.floor((current_length + 2 * padding - kernel_size) / stride + 1)
        
        # print(f"padding{padding}")
        # print(f"kernel_size{kernel_size}")
        # print(f"stride{stride}")
        # print(f"current_length after conv1{current_length}")
        
   
        # Apply max-pooling if indicated by the corresponding flag
        if max_pool_flags[idx] == 1:
            # print(current_length)
            maxpool_kernel, maxpool_stride, maxpool_padding, maxpool_dilation = max_pool_params
            current_length = math.floor((current_length + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel - 1) - 1) / maxpool_stride + 1)

            # print(f"maxpool_padding{maxpool_padding}")
            # print(f"maxpool_kernel{maxpool_kernel}")
            # print(f"maxpool_stride{maxpool_stride}")
            # print(f"maxpool_dilation{maxpool_dilation}")
            # print(f"current_length after maxPool{current_length}")
            
            
        # Append the output dimension after max-pooling (if applied)
        output_dims.append(current_length * out_channels)
        # print(f"{current_length}  * { out_channels}")
    return output_dims

def calculate_late_in_dim(n_data, conv_layers, maxpool_size, exit_placement, data_perc, max_pool_flags):
    """
    Calculate the late input dimensions for each layer in a neural network,
    optionally followed by a max-pooling operation.

    Parameters:
    - n_data: int, initial length of the data entering the first layer.
    - conv_layers: list of tuples, each containing (in_channels, out_channels, kernel_size, stride, padding)
                   for each convolutional layer.
    - maxpool_size: tuple, parameters for the max-pooling layers (kernel_size, stride, padding, dilation).
    - exit_placement: list of ints, indicating the layers at which exits are placed.
    - data_perc: list of floats, indicating the percentage of new data used at each exit.
    - max_pool_flags: list of int, flags (0 or 1) indicating whether a max-pooling layer follows each convolutional layer.

    Returns:
    - late_block_output_channels: list of output channels for late input blocks.
    - maxpool_kernel_size_list: list of max-pooling parameters for late input blocks.
    - output_dims: list of output dimensions after each layer.
    - data_ids: array of data indices used at each exit.
    """
    # Initialize lists to store the output dimensions, channels for late blocks, and maxpool kernel sizes
    output_dims = []
    late_block_output_channels = []
    maxpool_kernel_size_list = []
    
    count = 1 
    
    # Calculate input data for each layer based on the given data percentages
    n_data_each_input = np.floor((n_data * np.asarray(data_perc)) / 100)
    
    # Calculate cumulative data used and remaining data
    cumulative_data_used = np.sum(n_data_each_input)
    
    remaining_data = n_data - cumulative_data_used
    
    n_data_each_input = np.append(n_data_each_input, remaining_data)
    
    # Initialize data IDs for each layer
    data_ids = np.zeros((len(exit_placement) + 1, 2), dtype=int)
    id_begin = 0
    data_used = 0
    
    # Populate data IDs based on the data used
    for i in range(len(n_data_each_input)):
        
        data_used += n_data_each_input[i]
        
        data_ids[i, 0] = int(id_begin)
        data_ids[i, 1] = int(data_used)
        
        id_begin = data_used

    # print("Initial Data IDs:", data_ids)
    
    # Initialize the current length to the first input data length
    current_length = n_data_each_input[0]
    
    # print("Initial Current Length:", current_length)
    
    exit_id = 1
    
    # Iterate over each convolutional layer
    for idx, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
        # print(f"Layer {idx + 1} Conv Parameters: InChannels={in_channels}, OutChannels={out_channels}, KernelSize={kernel_size}, Stride={stride}, Padding={padding}")
    
        # May be Kernel size -1  
        # current_length = math.floor((current_length + 2 * padding - kernel_size) / stride + 1)
        current_length = math.floor((current_length + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        # print(f"Layer {idx + 1} Conv Output Length Calculation: ({current_length} + 2 * {padding} - {kernel_size}-1)-1 / {stride} + 1")
        # print(f"Layer {idx + 1} Conv Output Length:", current_length)
        
    
        # Apply max-pooling if indicated by the corresponding flag
        if max_pool_flags[idx] == 1:
            maxpool_kernel, maxpool_stride, maxpool_padding, maxpool_dilation = maxpool_size
            current_length = math.floor((current_length + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel - 1) - 1) / maxpool_stride + 1)
            # print(f"Layer {idx + 1} MaxPool Output Length Calculation: ({current_length} + 2 * {maxpool_padding} - {maxpool_dilation} * ({maxpool_kernel} - 1) - 1) / {maxpool_stride} + 1")
            # print(f"Layer {idx + 1} MaxPool Output Length:", current_length)
        
        if (current_length <= 0):
            return -1,-1,-1,0*data_ids
          
        # Append the output dimension after max-pooling (if applied)
        output_dims.append(out_channels * current_length)
        # print(f"Layer {idx + 1} Output Dimension:", out_channels * current_length)
        
      
        # Check if the current layer is an exit placement
        if count in exit_placement:
            late_block_output_channels.append(out_channels)
            
            # Calculate the new input length for the next layer
            new_input_length = n_data_each_input[exit_id]
            
           
            maxpool_kernel_new = int(np.floor(new_input_length / current_length))
            
            if (maxpool_kernel_new <= 1):
                # print(f"Invalid maxpool_kernel_new at exit {exit_id}: {maxpool_kernel_new}")
                return -1, -1, -1, 0 * data_ids
            
            # Initialize maxpool parameters for the new input
            maxpool_s = maxpool_kernel_new
            maxpool_d = 1
            maxpool_p = 0
            
            # Calculate the output length after the new max-pooling layer
            new_out_check = math.floor((new_input_length + 2 * maxpool_p - maxpool_d * (maxpool_kernel_new - 1) - 1) / maxpool_s + 1)
            
            if new_out_check == current_length:
                maxpool_kernel_size_list.append((maxpool_kernel_new, maxpool_d, maxpool_p))
            elif new_out_check < current_length:
                maxpool_p = (maxpool_kernel_new * current_length - new_input_length + maxpool_d * maxpool_s - maxpool_d + 1 - maxpool_kernel_new)
                maxpool_p = int(np.floor(maxpool_p))
                if maxpool_p > maxpool_kernel_new / 2:
                    maxpool_p = int(np.floor(maxpool_kernel_new / 2))
                new_out_check = math.floor((new_input_length + 2 * maxpool_p - maxpool_d * (maxpool_kernel_new - 1) - 1) / maxpool_s + 1)
                maxpool_kernel_size_list.append((maxpool_kernel_new, maxpool_d, maxpool_p))
            elif new_out_check > current_length:
                maxpool_d = (new_input_length - maxpool_kernel_new * current_length - 1 + maxpool_s) / (maxpool_s - 1)
                maxpool_d = int(np.floor(maxpool_d))
                new_out_check = math.floor((new_input_length + 2 * maxpool_p - maxpool_d * (maxpool_kernel_new - 1) - 1) / maxpool_s + 1)
                maxpool_kernel_size_list.append((maxpool_kernel_new, maxpool_d, maxpool_p))
                
            # print(f"Exit {exit_id} New Input Length: {new_input_length}, MaxPool Kernel Size: {maxpool_kernel_new}")
            
            current_length *= 2
            exit_id += 1
            
        count += 1
    
    # print("Final Output Dimensions:", output_dims)
    # print("Final Data IDs:", data_ids)
    
    return late_block_output_channels, maxpool_kernel_size_list, output_dims, data_ids


