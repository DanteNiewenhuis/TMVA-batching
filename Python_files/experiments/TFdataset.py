# %%

import tensorflow as tf

class Base:

    def __init__(self, inp):
        self.data = inp

    def __iter__(self):
        
        print("Iter")

        return self

    def __call__(self):

        self.__iter__()

        while(self.HasData()):

            outp = self.data[0]
            self.data = self.data[1:]

            yield outp 

    def HasData(self):
        if len(self.data) > 0:
            return True

        return False



# %%

class wrapper:

    def __init__(self, base:Base):
        self.base = base

    def __call__(self):
        self.base.__iter__()
        
        while(self.base.HasData()):
            yield self.base.__next__()


# %%

base = Base(range(1000))
w = wrapper(base)

ds = tf.data.Dataset.from_generator(w, output_types = (tf.int64)).batch(64)

# %%

for item in ds:
    print(item)
# %%

base = Base(range(10))
# %%

for item in base():
    print(item)

# %%
