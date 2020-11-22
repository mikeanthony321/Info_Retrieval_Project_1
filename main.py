import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt


# Figure 1: Class label distribution
# Figure 3: Class label distribution after cleaning
# Figure 4: Class label distribution after minority oversampling
def generateClassDistribution(class_labels, title, c1, c2):
    no_default_count, default_count = len(class_labels.loc[class_labels == 0]), len(class_labels.loc[class_labels == 1])

    plt.figure()
    plt.title(title)
    plt.bar(['No Default ({})'.format(no_default_count), 'Default ({})'.format(default_count)],
            height=[no_default_count, default_count],
            color=[c1, c2])


# 1: LOADING THE DATA
full_data = pd.read_csv('CreditCardClientDefault.csv')

# Generate Figure 1
generateClassDistribution(full_data['default payment next month'], 'Figure 1: Distribution of Default', (0, 0, 0.33), (0.33, 0, 0))

# 2: FEATURE SELECTION/REDUCTION
reduced_data = full_data.drop(columns=['ID', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE'])
reduced_data = reduced_data.loc[(reduced_data['PAY_1'] != -2) &
                                (reduced_data['PAY_2'] != -2) &
                                (reduced_data['PAY_3'] != -2) &
                                (reduced_data['PAY_4'] != -2) &
                                (reduced_data['PAY_5'] != -2) &
                                (reduced_data['PAY_6'] != -2) &
                                # Remove outlier
                                (reduced_data['LIMIT_BAL'] < 600000)]
reduced_data_array = reduced_data.to_numpy()

avg_monthly_bill = np.mean(reduced_data_array[:, 7:13], axis=1)
avg_monthly_payment = np.mean(reduced_data_array[:, 13:19], axis=1)

reduced_data_array = np.hstack([reduced_data_array[:, :7], avg_monthly_bill[:, np.newaxis], avg_monthly_payment[:, np.newaxis], reduced_data_array[:, 19, np.newaxis]])
reduced_data = pd.DataFrame(reduced_data_array, columns=['LIMIT_BAL', 'PAY_1', 'PAY_2', 'PAY_3',
                                                         'PAY_4', 'PAY_5', 'PAY_6', 'AVG_BILL_AMT',
                                                         'AVG_PAY_AMT', 'default payment next month'])

reduced_class_labels, reduced_data = reduced_data['default payment next month'], reduced_data.drop(columns=['default payment next month'])

# Generate Figure 3
generateClassDistribution(reduced_class_labels, 'Figure 3: Distribution After Cleaning', (0, 0, 0.67), (0.67, 0, 0))

# 3: DATA MANIPULATION
final_data, final_class_labels = SMOTENC(categorical_features=[1, 2, 3, 4, 5, 6], sampling_strategy=0.5).fit_resample(reduced_data, reduced_class_labels)

# Generate Figure 4
generateClassDistribution(final_class_labels, 'Figure 4: Distribution After SMOTE-NC', (0, 0, 1), (1, 0, 0))

# 4: MODEL PREDICTION
prediction = LogisticRegression(solver='sag', max_iter=10000).fit(final_data, final_class_labels)
probabilities = prediction.predict_proba(final_data)
predicted_classes = prediction.predict(final_data)

print('Model Accuracy: {}'.format(accuracy_score(predicted_classes, final_class_labels)))


# Figure 6: Probability of Default Given Granted Credit
# Figure 7: Probability of Default Given Amount Unspent Each Month
# Figure 8: Probability of Default Given Each Month's Payment Status
def generateProbabilityGraph(attribute_data, x_label, title, color, isForPayColumns):
    att = np.array(attribute_data)
    prob = np.array(probabilities[:, 1])

    if isForPayColumns:
        concat = np.hstack([att, prob[:, np.newaxis]])

        plt.figure(figsize=(14, 8))
        for i in range(6):
            heights = []
            for value in range(-1, 9):
                subset = concat[concat[:, i] == value][:, 6]
                heights.append(subset.mean() if len(subset) > 0 else 0)
            plt.subplot(2, 3, i + 1)
            plt.title('PAY_{}'.format(i + 1))
            plt.ylabel('Avg. Probability')
            plt.bar(np.arange(-1, 9), height=heights, color=color)
    else:
        plt.figure()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('Probability')
        plt.scatter(att, prob, color=color)


# 5: VISUALIZATION
visual_frame = pd.DataFrame(PCA(n_components=2, random_state=5).fit_transform(final_data), columns=['ATTR-1', 'ATTR-2'])

# Generate Figure 5
plt.figure()
plt.title('Figure 5: PCA-Reduced Class Relationship')
plt.scatter(x=visual_frame['ATTR-1'], y=visual_frame['ATTR-2'], c=probabilities[:, 1])

# Generate Figure 6
generateProbabilityGraph(final_data['LIMIT_BAL'],
                         'Total Credit',
                         'Figure 6: Probability of Default Given Total Credit',
                         (0.6, 0.3, 0),
                         False)
# Generate Figure 7
generateProbabilityGraph(final_data['LIMIT_BAL'] - final_data['AVG_BILL_AMT'],
                         'Avg. Unspent Each Month',
                         'Figure 7: Probability of Default Given Average Amount Unspent',
                         (0.7, 0.7, 0),
                         False)
# Generate Figure 8
generateProbabilityGraph(final_data.to_numpy()[:, 1:7], '', '', (0.4, 0, 0.4), True)

plt.show()
