import os



class PathLoader(object):
    def __init__(self, path):
        self.path = path 
        directories = os.listdir(path)
        
        self.directories = []
        
        for s in directories:
            if os.path.isdir(path + '/' + s):
                self.directories.append(s)
            
        self.directories.sort()
        
        
        self.pathes = {k:os.listdir(self.path +  f'/{k}') for k in self.directories}
        for k in self.pathes.keys():
            self.pathes[k].sort()
        self._len = sum(len(v) for _, v in self.pathes.items())
    
    def __len__(self):
        return self._len
    
    def __iter__(self): 
        for s in self.directories:
            directory = self.pathes[s]
            for i, v in enumerate(directory):
                is_last = (i != len(directory) -1 )
                
                yield self.path + f'/{s}' + f'/{v}', is_last
    
               
