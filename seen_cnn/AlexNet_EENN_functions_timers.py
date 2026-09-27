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
import time  # ADDED FOR TIMING
# from torchvision import models
import os

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


class AlexNetPartial(nn.Module):
    
    def __init__(self, in_channels, out_classes, input_sequence_length, thresholds, num_exits, is_training, n_data, exit_placement, new_data_perc):
        
        super(AlexNetPartial, self).__init__()
        
        self.is_training = is_training
        self.num_exits = num_exits + 1
        self.exit_placements = exit_placement #[1,3]
        self.n_data = n_data
        self.thresholds = thresholds 
        self.early_exits = nn.ModuleList()
        self.late_input = nn.ModuleList()
        self.out_classes = out_classes
        self.new_data_perc = new_data_perc
        
        # Calculate cumulative data percentage (matching SEEN version)
        data_used_partial = np.sum(self.new_data_perc)
        self.cumulative_data_percentage = list(self.new_data_perc) + [100 - data_used_partial]
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

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
        )
        
        conv_layers = [
        (in_channels, 64, 11, 4, 2),  
        (64, 192,5, 1,2),  
        (192, 384,3,1, 1), 
        (384, 256,3, 1,1),
        (256, 256,3, 1,1)
        ]
        
        maxpool_layers = [1, 1, 0, 0, 1]
        maxpool_size = (self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding, self.maxpool.dilation)
        
        self.output_channelsL, self.kernel_sizeL, self.EE_dims, self.data_ids = calculate_late_in_dim(n_data, conv_layers, maxpool_size, exit_placement, new_data_perc, maxpool_layers)
        
        if self.output_channelsL == -1:
            return
        
        feature_length = self.EE_dims[-1]         
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_length, 4096),
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
                out_conv_channel = self.output_channelsL[th]
                max_kernel_size = self.kernel_sizeL[th]
                late_data_input = LateDataInput(in_channels, out_conv_channel, max_kernel_size)
                self.late_input.append(late_data_input)
                th += 1

    def forward(self, x,  is_training):
        if is_training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)

    def forward_train(self, x):
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
        Forward inference with timing measurements for each exit point.
        Returns results and timing information.
        """
        All_x = copy.deepcopy(x)
        exit_id = 0
        cumulative_data_used = 0
        data_ids = self.data_ids
        x_partial = x[:, :, data_ids[exit_id, 0]:self.data_ids[exit_id, 1]]
        
        results = []
        
        # ADDED: Timer variables
        times = []    # List to store timing for each exit
        start_time_arr = [] #get the real time for the board
        start_time = time.time()
        start_time_arr.append(start_time)
        
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
                
                # ADDED: Record time for this exit
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                
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
                
                if exit_output['is_confident'].all():  # If all samples are confident, stop inference
                    return results, times, start_time_arr
                
                exit_id += 1
                cumulative_data_used = cumulative_data_used + self.cumulative_data_percentage[exit_id]
                new_sensor_block = All_x[:, :, data_ids[exit_id,0]:data_ids[exit_id,1]]
                
                # ADDED: Start timing for next section
                start_time = time.time()
                start_time_arr.append(start_time)
                
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
                times.append(0)  # ADDED: No timing for non-exit layers

        results.pop()
        times.pop()  # ADDED: Remove last non-exit entry
       
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        final_output = self.classifier(x)
        _, final_label = torch.max(final_output.data, 1)
       
        batch_size = np.size(final_label.detach().cpu().numpy())
       # Record the final output
        
        # ADDED: Record time for final classifier
        final_time = time.time() - start_time
        times.append(final_time)
        
        results.append({
           'output': final_label.detach().cpu().numpy(),  # Convert tensor to NumPy array
           'exit_point': np.ones(batch_size, dtype=int)*layer_num,
            'early_exit': np.ones(batch_size, dtype=int)*0,
           'confidence': np.ones(batch_size, dtype=int)*-1,  # Final exit does not have an associated confidence
            'threshold_value': np.ones(batch_size)*-1,
           'is_confident': np.ones(batch_size)*True, # Final output is always 'confident' as it's the last resort
           'partial_percentage': np.ones(batch_size, dtype=int)*cumulative_data_used

        })
        start_time_arr.append(time.time())  # ADDED: Record final time
        
        return results, times, start_time_arr  # MODIFIED: Return timing data

    # ADDED NEW METHOD: Batch inference wrapper
    def inference(self, test_loader, device):
        """
        Run inference on test data with timing measurements.
        Matches SEEN version interface.
        
        Args:
            test_loader: Array-like of shape [n_samples, channels, n_data] or Tensor
            device: torch.device to run inference on
        
        Returns:
            results: List of lists, where each inner list contains exit results for one sample
            times: List of lists, where each inner list contains timing for one sample
            start_time_arr_all: List of start times for each sample
        """
        self.eval()  # Switches the model to evaluation mode
        self.to(device)
        results = []
        timers = []
        start_time_arr_all = []
        
        # Handle both tensor and array inputs (matching SEEN version)
        if isinstance(test_loader, torch.Tensor):
            test_data = test_loader
        else:
            test_data = torch.tensor(test_loader, dtype=torch.float32)
        
        test_data = test_data.to(device)
        
        with torch.no_grad():
            for w in range(0, len(test_data)):
                temp = test_data[w,:,:]
                # Convert GPU tensor to CPU numpy before expanding dimensions
                if isinstance(temp, torch.Tensor):
                    temp_np = temp.cpu().numpy()
                else:
                    temp_np = temp
                inputs = torch.tensor(np.expand_dims(temp_np, axis=0), dtype=torch.float32, device=device)
                outputs, times, start_time_arr = self.forward(inputs, self.is_training) 
                results.append(outputs)
                timers.append(times)
                start_time_arr_all.append(start_time_arr)
                
        return results, timers, start_time_arr_all

    def train_model(self, train_loader, test_loader, loss_weights, num_epochs):
        """
        Trains the model with early exit blocks inserted and returns training and testing data.
        Matches the train_model signature from Alex_Net_functions.py
        """
        # Define optimizer and loss function
        optimizer = Adam(self.parameters(), lr=0.001)
        criterion = CrossEntropyLoss()

        # Move model to the device
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


# AlexNetEENN (non-sensor-aware) with timers
class AlexNetEENN(nn.Module):
    """
    AlexNet with early exits (non-sensor-aware version) with timing measurements.
    This is the non-sensor-aware version - no partial data sampling, no late input blocks.
    """
    
    def __init__(self, in_channels, out_classes, input_sequence_length, thresholds, num_exits, is_training, n_data, exit_placement):
        
        super(AlexNetEENN, self).__init__()
        
        self.is_training = is_training
        self.num_exits = num_exits + 1
        self.exit_placements = exit_placement
        self.n_data = n_data
        self.thresholds = thresholds 
        self.early_exits = nn.ModuleList()
        self.out_classes = out_classes
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=4, padding=2),
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
        )
        
        conv_layers = [
            (in_channels, 64, 11, 4, 2),  
            (64, 192, 5, 1, 2),  
            (192, 384, 3, 1, 1), 
            (384, 256, 3, 1, 1),
            (256, 256, 3, 1, 1)
        ]
        
        maxpool_layers = [1, 1, 0, 0, 1]
        maxpool_size = (self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding, self.maxpool.dilation)
        
        self.EE_dims = calculate_output_dim(n_data, conv_layers, maxpool_size, maxpool_layers)
        feature_length = self.EE_dims[-1]         

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_length, 4096),
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
                exit_in_dimension = self.EE_dims[idx-1]
                threshold = thresholds[th]
                exit_block = EarlyExitBlock(exit_in_dimension, out_classes, threshold)
                self.early_exits.append(exit_block)
                th += 1

    def forward(self, x, is_training):
        if self.is_training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)

    def forward_train(self, x):
        exit_id = 0
        exits_outputs = []
        
        layer_id = 1
        out = self.layer1(x)
        if layer_id in self.exit_placements:            
            exits_outputs.append(self.early_exits[exit_id](out)['logits'])
            exit_id += 1
        
        layer_id = layer_id + 1
        out = self.layer2(out)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](out)['logits'])
            exit_id += 1

        layer_id = layer_id + 1
        out = self.layer3(out)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](out)['logits'])
            exit_id += 1

        layer_id = layer_id + 1
        out = self.layer4(out)
        if layer_id in self.exit_placements:
            exits_outputs.append(self.early_exits[exit_id](out)['logits'])
            exit_id += 1

        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        
        final_output = self.classifier(out)
        exits_outputs.append(final_output)

        return exits_outputs
    
    def forward_inference(self, x):
        """
        Forward inference with timing measurements for each exit point.
        Returns results and timing information.
        Matches CNN1D_extended_EENN structure from EENN_functions_timers.py
        """
        results = []
        times = []  # Track time for each exit
        idx = 0
    
        operations = [
            (self.layer1, 1),
            (self.layer2, 2),
            (self.layer3, 3),
            (self.layer4, 4),
            (self.layer5, 5)
        ]
        
        # Start timing for first layer
        start_time_arr = [] #get the real time for the board
        start_time = time.time()
        start_time_arr.append(start_time)
    
        for operation, layer_num in operations:
            x = operation(x)
            
            if layer_num in self.exit_placements and layer_num != 5:
                # Process the early exit
                exit_output = self.early_exits[idx](x)
                
                _, exit_label = torch.max(exit_output['logits'].data, 1)
                batch_size = np.size(exit_label.detach().cpu().numpy())
                
                # Record time for this exit (from start of this section)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
    
                results.append({
                    'output': exit_label.detach().cpu().numpy(),
                    'exit_point': np.ones(batch_size, dtype=int)*layer_num,
                    'early_exit': np.ones(batch_size, dtype=int)*1,
                    'threshold_value': np.ones(batch_size)*self.early_exits[idx].threshold,
                    'confidence': exit_output['entropy'].detach().cpu().numpy(),
                    'is_confident': exit_output['is_confident'].cpu().numpy()
                })
                
                # Start timing for next section
                start_time = time.time()
                start_time_arr.append(start_time)
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
                times.append(0)  # No timing for non-exit layers
        
        results.pop()
        times.pop()  # Remove last non-exit entry
    
        x = x.view(x.size(0), -1)  # Flatten
        
        final_output = self.classifier(x)
        _, final_label = torch.max(final_output.data, 1)
        batch_size = np.size(final_label.detach().cpu().numpy())
        
        # Record time for final classifier
        final_time = time.time() - start_time
        times.append(final_time)
    
        results.append({
            'output': final_label.detach().cpu().numpy(),
            'exit_point': np.ones(batch_size, dtype=int)*layer_num,
            'early_exit': np.ones(batch_size, dtype=int)*0,
            'confidence': np.ones(batch_size, dtype=int)*-1,
            'threshold_value': np.ones(batch_size)*-1,
            'is_confident': np.ones(batch_size)*True
        })
        start_time_arr.append(time.time())
    
        return results, times, start_time_arr

    def inference(self, test_loader, device):
        """
        Run inference on test data with timing measurements.
        Matches SEEN version interface.
        
        Args:
            test_loader: Array-like of shape [n_samples, channels, n_data] or Tensor
            device: torch.device to run inference on
        
        Returns:
            results: List of lists, where each inner list contains exit results for one sample
            times: List of lists, where each inner list contains timing for one sample
            start_time_arr_all: List of start times for each sample
        """
        self.eval()  # Switches the model to evaluation mode
        self.to(device)
        results = []
        timers = []
        start_time_arr_all = []
        
        # Handle both tensor and array inputs (matching SEEN version)
        if isinstance(test_loader, torch.Tensor):
            test_data = test_loader
        else:
            test_data = torch.tensor(test_loader, dtype=torch.float32)
        
        test_data = test_data.to(device)
        
        with torch.no_grad():
            for w in range(0, len(test_data)):
                temp = test_data[w,:,:]
                # Convert GPU tensor to CPU numpy before expanding dimensions
                if isinstance(temp, torch.Tensor):
                    temp_np = temp.cpu().numpy()
                else:
                    temp_np = temp
                inputs = torch.tensor(np.expand_dims(temp_np, axis=0), dtype=torch.float32, device=device)
                outputs, times, start_time_arr = self.forward(inputs, self.is_training) 
                results.append(outputs)
                timers.append(times)
                start_time_arr_all.append(start_time_arr)
                
        return results, timers, start_time_arr_all

    def train_model(self, train_loader, test_loader, loss_weights, num_epochs):
        """
        Trains the model with early exit blocks inserted and returns training and testing data.
        Matches the train_model signature from Alex_Net_functions.py
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


# Keep the original helper functions unchanged
def calculate_output_dim(n_data, conv_layers, maxpool_size, maxpool_layers):
    """
    Calculate the output dimensions for each layer in a neural network,
    optionally followed by a max-pooling operation.
    """
    output_dims = []
    current_length = n_data
    
    maxpool_kernel, maxpool_stride, maxpool_padding, maxpool_dilation = maxpool_size
    
    for idx, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
        
        current_length = math.floor((current_length + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        
        if maxpool_layers[idx] == 1:
            current_length = math.floor((current_length + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel - 1) - 1) / maxpool_stride + 1)
            
        output_dims.append(current_length * out_channels)
        
    return output_dims


def calculate_late_in_dim(n_data, conv_layers, maxpool_size, exit_placement, data_perc, max_pool_flags):
    """
    Calculate the late input dimensions for each layer in a neural network,
    optionally followed by a max-pooling operation.
    """
    output_dims = []
    late_block_output_channels = []
    maxpool_kernel_size_list = []
    
    count = 1 
    
    n_data_each_input = np.floor((n_data * np.asarray(data_perc)) / 100)
    
    cumulative_data_used = np.sum(n_data_each_input)
    
    remaining_data = n_data - cumulative_data_used
    
    n_data_each_input = np.append(n_data_each_input, remaining_data)
    
    data_ids = np.zeros((len(exit_placement) + 1, 2), dtype=int)
    id_begin = 0
    data_used = 0
    
    for i in range(len(n_data_each_input)):
        
        data_used += n_data_each_input[i]
        
        data_ids[i, 0] = int(id_begin)
        data_ids[i, 1] = int(data_used)
        
        id_begin = data_used
    
    current_length = n_data_each_input[0]
    
    exit_id = 1
    
    for idx, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
    
        current_length = math.floor((current_length + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        
        if max_pool_flags[idx] == 1:
            maxpool_kernel, maxpool_stride, maxpool_padding, maxpool_dilation = maxpool_size
            current_length = math.floor((current_length + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel - 1) - 1) / maxpool_stride + 1)
        
        if (current_length <= 0):
            return -1,-1,-1,0*data_ids
          
        output_dims.append(out_channels * current_length)
        
        if count in exit_placement:
            late_block_output_channels.append(out_channels)
            
            new_input_length = n_data_each_input[exit_id]
            
            maxpool_kernel_new = int(np.floor(new_input_length / current_length))
            
            if (maxpool_kernel_new <= 1):
                return -1, -1, -1, 0 * data_ids
            
            maxpool_s = maxpool_kernel_new
            maxpool_d = 1
            maxpool_p = 0
            
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
            
            current_length *= 2
            exit_id += 1
            
        count += 1
    
    return late_block_output_channels, maxpool_kernel_size_list, output_dims, data_ids