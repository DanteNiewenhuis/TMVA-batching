from threading import Thread
import threading
import time

def f(name):
    time.sleep(2)    
    print('hello', name)
    
class C:

    def f(self, name):
        time.sleep(2)    
        print('hello', name)


    def start_thread(self, name):
        self.p = Thread(target=self.f, args=(name,))
        print("made Thread")
        self.p.start()
        print("started Thread")


    def end_thread(self):
        print("wait for Thread")
        self.p.join()
        print("Thread done")

if __name__ == '__main__':
    print(f"threads before: {threading.active_count()}")

    t = Thread(target=f, args=("bob",))
    t.start()

    print(f"threads after: {threading.active_count()}")

    time.sleep(3)

    print(f"threads after time: {threading.active_count()}")

    t.join()

    print(f"threads after join: {threading.active_count()}")

    