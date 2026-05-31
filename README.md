# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
To evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization.

## SCENARIO:
You are part of a content curation team for an educational platform that delivers quick summaries of research papers to undergraduate students. Your task is to summarize a 500-word technical article on "The Basics of Blockchain Technology" using multiple AI platforms and prompting strategies.

Your goal is to determine which combination of prompting technique + platform provides the best summary in terms of:

Accuracy
Coherence
Simplicity
Speed
User experience
## PROGRAM AND OUTPUT
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
print("First 5 rows:")
display(df.head())
```
<img width="741" height="255" alt="image" src="https://github.com/user-attachments/assets/72915bc9-2384-4f2b-a569-532975b88403" />
<img width="529" height="522" alt="image" src="https://github.com/user-attachments/assets/33b033f5-9d83-4822-b9d4-ba375071d98e" />

```py
print("\nDataset info:")dfinfo()
print("\nStatistical Summary:") display(df.describe())
```
<img width="686" height="355" alt="image" src="https://github.com/user-attachments/assets/dbdf6c8b-ec32-49fa-9d93-3eea9e19f550" />

```py
print("\nMissing Values:")
```

<img width="263" height="396" alt="image" src="https://github.com/user-attachments/assets/b2213bfc-5413-429e-b241-d4f0d5ba951b" />

```py
plt.figure() df['age'].hist(bins=20) plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
```
<img width="726" height="575" alt="image" src="https://github.com/user-attachments/assets/8277b461-bef1-423d-9be3-f6a42a8e6b60" />

```py
plt.figure()
sns.countplot(x='survived', data=df) plt.title("Survival Count") plt.show()
```

<img width="825" height="413" alt="image" src="https://github.com/user-attachments/assets/15f7f127-a19e-442b-a725-387fedd64949" />

```py
plt.figure() sns.boxplot(x='survived', y='age', data=df) plt.title("Age vs Survival") plt.show()
```
<img width="766" height="600" alt="image" src="https://github.com/user-attachments/assets/52e6b75e-e566-4089-93ee-0f6ba93664a2" />

```py
plt.figure()
sns.scatterplot(x='age', y='fare', data=df) plt.title("Age vs Fare") plt.show()
```
<img width="748" height="560" alt="image" src="https://github.com/user-attachments/assets/f0924827-2e84-48fb-88cf-074b4f2a5077" />

```py
sns.pairplot(df[['age', 'fare', 'survived']]) plt.show()
```
<img width="743" height="746" alt="image" src="https://github.com/user-attachments/assets/28c71ba4-f5c8-456c-b223-780de3e5f50e" />

```py
plt.figure()
corr = df.corr(numeric_only=True) sns.heatmap(corr, annot=True) plt.title("Correlation Matrix") plt.show() print("\nEDA Completed Successfully!")
```
<img width="788" height="691" alt="image" src="https://github.com/user-attachments/assets/c6ff950f-8ce5-4b26-86f5-353b0d651462" />


## RESULT
Exploratory Data analysis performed in jupiter notebook successfully
