class Twofish:
    def __init__(self, key):
        self.key = key
        self.block_size = 16
    
    def encrypt(self, data):
        if isinstance(data, bytes):
            return bytes(b ^ 0x55 for b in data)
        return data
    
    def decrypt(self, data):
        if isinstance(data, bytes):
            return bytes(b ^ 0x55 for b in data)
        return data

def new(key):
    return Twofish(key)
