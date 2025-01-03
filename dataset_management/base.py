

class WWADLBase():
    def __init__(self):
        self.data = None
        self.label = None


    def load_data(self, file_path):
        pass

    def show_info(self):
        print(self.data.shape)
        print(self.label)