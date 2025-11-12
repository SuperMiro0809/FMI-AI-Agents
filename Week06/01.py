import seaborn as sms
from matplotlib import pyplot as plt

if __name__ == "__main__":
    df = sms.load_dataset('penguins')

    sms.pairplot(df, hue='species')
    plt.show()