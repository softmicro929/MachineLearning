class DataSets:
    train_data = None
    validate_data = None
    test_data = None

    def __init__(self,train_data,validate_data,test_data):
        DataSets.train_data = train_data
        DataSets.validate_data = validate_data
        DataSets.test_data = test_data
        pass

    class TrainData:
        def __init__(self,input_x,input_y):
            self.input_x = input_x
            self.input_y = input_y
            pass
        def get_x(self):
            return self.input_x
            pass
        def get_y(self):
            return self.input_y
            pass

    class ValidateData:
        def __init__(self,input_x,input_y):
            self.input_x = input_x
            self.input_y = input_y
            pass
        def get_x(self):
            return self.input_x
            pass
        def get_y(self):
            return self.input_y
            pass

    class TestData:
        def __init__(self,input_x,input_y):
            self.input_x = input_x
            self.input_y = input_y
            pass
        def get_x(self):
            return self.input_x
            pass
        def get_y(self):
            return self.input_y
            pass

