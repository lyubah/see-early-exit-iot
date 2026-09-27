import random
import pandas as pd

class Config:
    
    def __init__(self):
        self.depths =  [i for i in range(50,800,50)]
        self.treenum_options = [i for i in range(50,150)]  
        self.RESTRICTED_POINTS = []
        # num of exits !!!!!!
    
    def generate_random_split(self, min_pct=0.25,min_distance=0.1,restricted_zone=(0.2, 0.3),max_retries=1000):
        
        self.RESTRICTED_POINTS = []
        max_m = int(1 / min_pct)
        
        for _ in range(max_retries):
            m = random.randint(2, max_m)
            total_min = min_pct * m
            S = 1 - total_min

            # Generate random points and compute splits
            random_points = [random.uniform(0, S) for _ in range(m - 1)]
            random_points.sort()
            points = [0.0] + random_points + [S]
            intervals = [points[i+1] - points[i] for i in range(len(points)-1)]
            splits = [min_pct + interval for interval in intervals]

            # Build split points
            split_points = []
            current = 0.0
            for split in splits:
                current += split
                split_points.append(round(current, 2))
            split_points[-1] = 1.0  # Ensure last point is exactly 1.0

            # Check for duplicates
            if len(split_points) != len(set(split_points)):
                continue

            # Check restricted zone rules
            valid = True
            restricted_in_current = []
            for i, point in enumerate(split_points[:-1]):
                if restricted_zone[0] <= point < restricted_zone[1]:
                    next_point = split_points[i + 1]
                    if (next_point - point) < min_distance:
                        valid = False
                        break
                    restricted_in_current.append(point)
                
                for prev_point in self.RESTRICTED_POINTS:
                    if abs(point - prev_point) < min_distance:
                        valid = False
                        break

            if valid:
                self.RESTRICTED_POINTS.extend(restricted_in_current)
                return split_points
        
        raise ValueError("Failed to generate valid splits after max retries.")

    def select_random_depth(self):
        self.max_depth = random.choice(self.depths)

    def select_random_treenum(self ,deviation):
        divisible_options = [num for num in self.treenum_options if num % deviation == 0]
        self.n_estimators = int((random.choice(divisible_options)/deviation))

def generate_configurations_for_datasets(datasetnames , num_configurations=100):
    """
    Generates a list of configurations for various datasets.
    Each configuration is a dictionary including dataset_name, split_points, max_depth, and n_estimators.
    """
    all_configurations = []
    
    # Define your datasets
    # datasets = ['Shoaib', 'Epilepsy', 'EMGPhysical', 'SelfRegulationSCP1', 'WESADchest', 'PAMAP2']


    for dataset_name in datasetnames:
        print(f"Generating configurations for dataset: {dataset_name}")
        for _ in range(num_configurations):
            config_obj = Config()
            try:
                splits = config_obj.generate_random_split()
                num_exit = len(splits)
                config_obj.select_random_depth()
                config_obj.select_random_treenum(num_exit)
            except ValueError:
                continue  # Skip configurations with invalid splits
            
            config_data = {
                'dataset_name': dataset_name,
                'split_points': splits, # Store as list of floats
                'max_depth': config_obj.max_depth,
                'n_estimators': config_obj.n_estimators
            }
            all_configurations.append(config_data)
    
    print(f"Generated {len(all_configurations)} configurations across all datasets.")
    return all_configurations

# This part will only run if the script is executed directly, not when imported
if __name__ == '__main__':
    # Example usage: Generate 5 configurations per dataset
    configurations_list = generate_configurations_for_datasets(['Epilepsy'], num_configurations=2)
    
    # You can optionally save this list to a file if needed for debugging or external use,
    # but the primary goal is to return it as a list of dictionaries for parallel processing.
    # For demonstration, let's print the first few configs
    print("\nSample generated configurations (first 3):")
    for i, config in enumerate(configurations_list[:3]):
        print(f"Config {i+1}: {config}")