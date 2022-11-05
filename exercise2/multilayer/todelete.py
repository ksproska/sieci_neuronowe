import numpy as np


def main():
    labels = np.arange(0, 6)
    labels[0] = 2

    print(labels)
    y_hat = np.diag(np.ones(shape=(6,))) - np.diag(np.random.random(6)/10)
    print(y_hat)
    predictions = np.argmax(y_hat, axis=1)
    print(np.mean(predictions == labels))




if __name__ == '__main__':
    main()