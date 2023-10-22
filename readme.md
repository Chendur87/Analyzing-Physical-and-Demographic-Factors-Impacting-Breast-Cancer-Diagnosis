Breast cancer is expected to affect 1 in 8 women in the United States and is one of the most common types of cancer worldwide. The treatability of breast cancer depends on the cancer stage, with the earlier stages being more curable. However, many don’t realize that a cancerous tumor is developing, only finding out later and making the cancer harder to treat. Therefore, the motive of this project is to see if there are any trends in tumor size and other characteristics (density of clumping in cells, differentiation of cells, grade of cancer cells, age, etc.) in determining whether someone has breast cancer or is developing breast cancer in the earlier stages. From this project, cancer might not be cured, but we can be more educated about the factors that tend to affect the development of this cancer more so than others, hopefully utilizing the findings from this project to detect breast cancer quicker.

Additionally, this project heavily focuses on statistical analysis and utilizing ML to confirm significant differences between malignant and benign tumors. Taking skills from AP Statistics and applying them to data science allows for further knowledge about the math behind data science. Constructing confidence intervals and performing hypothesis testing on these breast cancer datasets helps foster a deeper understanding of how data can paint a picture of how features can vary between patients with malignant tumors and those who do not have cancer.

The research questions were the following:
1. To what extent do the features of tumors (size, uniformity, etc.) correlate to tumors being either malignant (cancerous) or benign (non-cancerous)?
2. How strong is the correlation between the demographic factors (age, race, marital status) of the patient and the features of the tumors (size, grade of tumor, percent carcinogenic)? How big of a role do factors outside of the tumor area correlate to factors within the tumor area?
3. How well can Machine Learning determine if a patient has breast cancer (malignant vs. benign tumors) based on the features of the tumor? Can we identify optimal hyperparameters to create an optimal model and determine the top features that contribute to the model's decisions?

My conclusions were the following:
1. The radius, perimeter, number of concave points, and area have the strongest evidence for a statistically significant difference in malignant and benign tumors. The texture, smoothness, and symmetry of the tumors have the weakest evidence for a statistically significant difference in malignant and benign tumors. The compactness and the concavity of the tumors has strong evidence of significant difference, but not as strong as the radius, perimeter, concave points, and area features.
2. For the most part, there was a weak to, at best, moderate correlation between the demographic factors and the features of the tumors. Demographic factors such as age, race, and marital status do not dictate the characteristics of the features of the tumor.
3. A RandomForestClassifier was trained on the feature datasets and had an above 90% testing accuracy in determining whether the patient had either a malignant or benign tumor. The top 3 features that
contributed to that model were the radius, perimeter, and number of concave points.

More information about my findings are in my report: https://docs.google.com/document/d/1yzAmNR6xWTbcu0D4UIknUUbauI_W6MiJV3mRjpG-iZI/edit

Here is a link to the datasets for this project: https://drive.google.com/drive/folders/11KRy_paR-_CxKfIR4TGdeDyH4aA41G8Z?usp=share_link
