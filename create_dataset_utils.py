import pickle


def dump(path_to_file, imgs,labels, ids, classes):
    with open(path_to_file,'wb') as file:
        pickle.dump([imgs,labels,ids, classes],file)

def load(path_to_file):
    with open(path_to_file,'rb') as file:
        imgs,labels,ids, classes = pickle.load(file)
    return imgs,labels,ids, classes
