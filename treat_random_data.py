from treat_data import treat_data
import numpy as np

class treat_random_data(treat_data):
    def __init__(self, output_file):
        super().__init__(output_file)  # Initialize the parent class
     
     
    '''
     input: select data
     output : change the order of permutation labels try to get a different set each time
     '''  
    def get_permutated_labels(self, selected_data, num_permutations): 
        unique_permutations = set() # only unique objects
        while len(unique_permutations) < num_permutations:
            labels = selected_data.iloc[:,(selected_data.shape[1]- 1)]
            permutated_labels = tuple(np.random.permutation(labels))
            unique_permutations.add(permutated_labels)
        
        #convert back to list
        permutated_labels_list = [list(p) for p in unique_permutations]
        
        return permutated_labels_list
        
    def _call_(self,model = None,n_repeats = None,num_permutations = None, sex = None, choice = None, hormones = None, architype = None):
         #select data to work with
         if choice == "2":
            selected_data = super().select_data(sex, choice, hormones)#from parent
         elif choice == "3":
            selected_data = super().select_data(sex, choice, hormones,architype)
        #create permutation
        # Set to track unique permutations
         permutated_labels_list = self.get_permutated_labels(selected_data, num_permutations)
         #including all permutations
         results_rand = {}
         for i,labels in  enumerate(permutated_labels_list,1):
             # add to select_data
             selected_data.iloc[:,(selected_data.shape[1]- 1)] = labels
             results_dict = self.train_learning(selected_data, model,n_repeats )
             results_rand[i] = results_dict
         return results_rand     
             
             
             
            