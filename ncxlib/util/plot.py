import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, auc):
    # print(fpr, tpr, auc)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()

def plot_2d(X, Y, title="Data Plot", alpha=0.6):
    """
    Plot 2D data points with labels.

    Parameters:
    - X: numpy array of shape (n_samples, 2), the data points.
    - Y: numpy array of shape (n_samples,), the labels (0 or 1).
    - title: string, the title of the plot.
    - alpha: float, transparency of the scatter points.
    """
    plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], label="Class 0", alpha=alpha)
    
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], label="Class 1", alpha=alpha)
    
    plt.legend()
    plt.title(title)
    
    plt.show()