# AnimeRs

An anime recommendation system.

The purpose of this project is to research and create an anime recommendation system.

This project was created with **Python (version 3.8.7), surprise, pandas, numpy and more libraries**.

## Project Research

In order to understand the steps and what we did you are welcome to look at [the research jupyter notebook](https://github.com/leorrose/AnimeRS/blob/main/research_notebook.ipynb).

We tested various recommender systems provided by surprise and these are the results we got:

### **User-Based CF**

| FIELD1                       | RMSE  | MSE   | MAE   | P@5   | R@5   | F1@5  | P@10  | R@10  | F1@10 | P@15  | R@15  | F1@15 |
|------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| SVD                          | 1.247 | 1.554 | 0.951 | 0.822 | 0.808 | 0.815 | 0.82  | 0.831 | 0.826 | 0.821 | 0.834 | 0.827 |
| SVDpp                        | 1.242 | 1.543 | 0.945 | 0.819 | 0.79  | 0.804 | 0.815 | 0.808 | 0.812 | 0.815 | 0.809 | 0.812 |
| SlopeOne                     | 1.486 | 2.209 | 1.136 | 0.771 | 0.684 | 0.725 | 0.77  | 0.697 | 0.731 | 0.775 | 0.701 | 0.736 |
| NMF                          | 2.165 | 4.686 | 1.87  | 0.282 | 0.141 | 0.188 | 0.286 | 0.143 | 0.19  | 0.28  | 0.141 | 0.188 |
| NormalPredictor              | 2.113 | 4.465 | 1.677 | 0.738 | 0.617 | 0.672 | 0.74  | 0.628 | 0.679 | 0.737 | 0.625 | 0.676 |
| KNNBaselineMSD               | 1.398 | 1.956 | 1.067 | 0.817 | 0.775 | 0.796 | 0.816 | 0.794 | 0.805 | 0.812 | 0.79  | 0.801 |
| KNNBaselineCosine            | 1.387 | 1.924 | 1.059 | 0.821 | 0.785 | 0.803 | 0.816 | 0.799 | 0.808 | 0.816 | 0.799 | 0.807 |
| KNNBaselinePearson           | 1.345 | 1.808 | 1.023 | 0.826 | 0.828 | 0.827 | 0.824 | 0.851 | 0.838 | 0.822 | 0.853 | 0.837 |
| KNNBaselinePearsonBaseline   | 1.362 | 1.854 | 1.038 | 0.825 | 0.825 | 0.825 | 0.823 | 0.848 | 0.835 | 0.823 | 0.847 | 0.835 |
| KNNBasicMSD                  | 1.57  | 2.466 | 1.197 | 0.812 | 0.792 | 0.802 | 0.808 | 0.81  | 0.809 | 0.808 | 0.812 | 0.81  |
| KNNBasicCosine               | 1.582 | 2.502 | 1.212 | 0.814 | 0.805 | 0.809 | 0.81  | 0.821 | 0.815 | 0.811 | 0.823 | 0.817 |
| KNNBasicPearson              | 1.599 | 2.558 | 1.247 | 0.812 | 0.885 | 0.847 | 0.812 | 0.925 | 0.865 | 0.812 | 0.925 | 0.865 |
| KNNBasicPearsonBaseline      | 1.612 | 2.598 | 1.251 | 0.811 | 0.877 | 0.843 | 0.811 | 0.913 | 0.859 | 0.811 | 0.916 | 0.86  |
| KNNWithMeansMSD              | 1.376 | 1.895 | 1.048 | 0.787 | 0.736 | 0.76  | 0.786 | 0.755 | 0.77  | 0.788 | 0.758 | 0.773 |
| KNNWithMeansCosine           | 1.356 | 1.839 | 1.03  | 0.787 | 0.738 | 0.762 | 0.786 | 0.758 | 0.772 | 0.786 | 0.758 | 0.772 |
| KNNWithMeansPearson          | 1.415 | 2.001 | 1.079 | 0.733 | 0.756 | 0.744 | 0.735 | 0.788 | 0.761 | 0.733 | 0.787 | 0.759 |
| KNNWithMeansPearsonBaseline  | 1.422 | 2.023 | 1.085 | 0.736 | 0.752 | 0.744 | 0.739 | 0.782 | 0.76  | 0.739 | 0.783 | 0.76  |
| KNNWithZscoremsd             | 1.379 | 1.901 | 1.038 | 0.792 | 0.745 | 0.768 | 0.788 | 0.762 | 0.775 | 0.791 | 0.763 | 0.777 |
| KNNWithZscoreCosine          | 1.353 | 1.831 | 1.019 | 0.795 | 0.751 | 0.772 | 0.792 | 0.77  | 0.781 | 0.792 | 0.771 | 0.782 |
| KNNWithZscorePearson         | 1.41  | 1.988 | 1.073 | 0.737 | 0.759 | 0.748 | 0.737 | 0.788 | 0.761 | 0.738 | 0.791 | 0.763 |
| KNNWithZscorePearsonBaseline | 1.423 | 2.025 | 1.082 | 0.741 | 0.756 | 0.748 | 0.738 | 0.782 | 0.76  | 0.741 | 0.786 | 0.763 |
| BaselineOnly                 | 1.27  | 1.612 | 0.971 | 0.832 | 0.85  | 0.841 | 0.828 | 0.874 | 0.85  | 0.829 | 0.878 | 0.853 |
| CoClustering                 | 1.312 | 1.721 | 1.006 | 0.788 | 0.733 | 0.76  | 0.785 | 0.75  | 0.767 | 0.787 | 0.75  | 0.768 |


### **Item-Based CF**

| FIELD1                       | RMSE  | MSE   | MAE   | P@5   | R@5   | F1@5  | P@10  | R@10  | F1@10 | P@15  | R@15  | F1@15 |
|------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| KNNBaselineMSD               | 1.366 | 1.866 | 1.026 | 0.791 | 0.748 | 0.769 | 0.793 | 0.773 | 0.783 | 0.791 | 0.772 | 0.781 |
| KNNBaselineCosine            | 1.34  | 1.795 | 1.01  | 0.794 | 0.755 | 0.774 | 0.791 | 0.775 | 0.783 | 0.792 | 0.778 | 0.785 |
| KNNBaselinePearson           | 1.376 | 1.893 | 1.038 | 0.813 | 0.793 | 0.803 | 0.811 | 0.817 | 0.814 | 0.809 | 0.814 | 0.811 |
| KNNBaselinePearsonBaseline   | 1.387 | 1.923 | 1.047 | 0.811 | 0.787 | 0.799 | 0.809 | 0.81  | 0.809 | 0.808 | 0.809 | 0.808 |
| KNNBasicMSD                  | 1.519 | 2.308 | 1.141 | 0.779 | 0.775 | 0.777 | 0.778 | 0.803 | 0.79  | 0.777 | 0.804 | 0.79  |
| KNNBasicCosine               | 1.513 | 2.291 | 1.14  | 0.776 | 0.784 | 0.78  | 0.776 | 0.815 | 0.795 | 0.773 | 0.812 | 0.792 |
| KNNBasicPearson              | 1.599 | 2.557 | 1.22  | 0.802 | 0.851 | 0.826 | 0.801 | 0.885 | 0.841 | 0.802 | 0.887 | 0.843 |
| KNNBasicPearsonBaseline      | 1.597 | 2.551 | 1.215 | 0.8   | 0.838 | 0.818 | 0.798 | 0.871 | 0.833 | 0.799 | 0.872 | 0.834 |
| KNNWithMeansMSD              | 1.38  | 1.905 | 1.041 | 0.796 | 0.732 | 0.763 | 0.794 | 0.75  | 0.772 | 0.793 | 0.75  | 0.771 |
| KNNWithMeansCosine           | 1.359 | 1.848 | 1.024 | 0.794 | 0.734 | 0.763 | 0.797 | 0.756 | 0.776 | 0.795 | 0.757 | 0.775 |
| KNNWithMeansPearson          | 1.466 | 2.148 | 1.114 | 0.809 | 0.765 | 0.786 | 0.807 | 0.782 | 0.794 | 0.807 | 0.783 | 0.795 |
| KNNWithMeansPearsonBaseline  | 1.471 | 2.165 | 1.116 | 0.805 | 0.754 | 0.778 | 0.808 | 0.777 | 0.792 | 0.805 | 0.776 | 0.79  |
| KNNWithZscoreMSD             | 1.386 | 1.922 | 1.043 | 0.799 | 0.735 | 0.766 | 0.8   | 0.758 | 0.778 | 0.799 | 0.757 | 0.777 |
| KNNWithZscoreCosine          | 1.364 | 1.86  | 1.026 | 0.801 | 0.742 | 0.77  | 0.8   | 0.763 | 0.781 | 0.8   | 0.762 | 0.78  |
| KNNWithZscorePearson         | 1.47  | 2.161 | 1.118 | 0.808 | 0.765 | 0.786 | 0.81  | 0.785 | 0.797 | 0.809 | 0.786 | 0.797 |
| KNNWithZscorePearsonBaseline | 1.471 | 2.165 | 1.117 | 0.808 | 0.764 | 0.785 | 0.806 | 0.779 | 0.792 | 0.806 | 0.779 | 0.793 |


## Project Setup and Run

1. Clone this repository.
2. Open cmd/shell/terminal and go to project folder: `cd AnimeRS`
3. Install project dependencies: `pip install -r requirements.txt`
4. Run the streamlit app: `streamlit run ./app/anime_app.py`
5. Enjoy the application.

Please let me know if you find bugs or something that needs to be fixed.

Hope you enjoy it
