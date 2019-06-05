import os
import pickle


def pickle_dump(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    a = list(range(10))
    print(a)
    demo_file = os.path.join(os.getcwd(), 'temp', 'pickle_demo.pkl')
    print(demo_file)
    pickle_dump(a, demo_file)
    new_a = pickle_load(demo_file)
    print(new_a)
