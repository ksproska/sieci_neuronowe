import numpy as np
import inspect


class Foo():
    def __init__(self):
        self.a = 1

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            inspect.getmembers(self)[2][1][k] = v


def main():
    f = Foo()
    print(f.a)
    # inspect.getmembers(f)[2][1]['a'] = 4
    print(f.a)
    f.__dict__['a'] = 4
    print(f.a)

    d1 = {"a": 1, "b": 2, "c":3}
    d2 = {"a": 2, "b": 2, "c": 2}

    for key in d1:
        if not d1[key] == d2[key]:
            print(d1[key], d2[key])



if __name__ == '__main__':
    main()
