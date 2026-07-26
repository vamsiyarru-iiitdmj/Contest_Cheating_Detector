# Contest Cheating Detector

## About the project:
Classifies a Kaggle ML regression contest (based on RMSE) participant into one of 9 categories (hardworker, benchmark_tweaker, daily_sprinter, consistent_late_breakthrough, low_interest_quitter, consistent_ai_leap, ai_dump_sincere_start, late_joiner_dangerous, pure_ai_quick_dump) using their raw submission history, an XGBoost classifier  and Decision Tree classifier (Mostly for visualization purpose).

## Origin of the Idea:
I had participated in a online contest held by our college's Programming Club. It was a week long ML regression based contest  in which the scores were based on RMSE. The hackathon started with a few members submitting their submissions after initial EDA. But a the days progressed we found many tricks (patterns in the data) , which were not enough to get a score below 0.48 rmse. But , I noticed people joined only a day before getting the score of 0.42, 0.43 etc. I was shocked , I thought there were other hidden in the dataset to find out and improve the score. I tried hard to find out the patterns, some more new feature interactions , etc. But the efforts barely moved my score. 
The competition ended. I was once in the 2nd place in the contest ended at the 22nd place. I felt like I was unable to perform in a beginner contest held to test our basic EDA and regression application skills. After the contest, I once tried out AI to check the patterns which I was unable to find out. I found that AI was using cyclic encoding, specifically selecting the polynomial features, and many other features which doesn't make any sense. But , when I submitted the csv given by AI , I was shocked it landed directly on 0.6 mark, which took more than 3 days for me to score like that. Then I just gave a simple prompt, like "you have scored - ..... , can you improve your performance ?" . Then the next csv it provided landed on 0.53. The sequence of scores looked like this : 0.60, 0.53, 0.45, 0.47, 0.42, 0.43, 0.425. By observing this pattern , I understood how one can improve his scores rapidly in just a few submissions. The scores are always improving with only a few ups(error) during the convergence towards minimum rmse.
Then I realised that the AI csv submitters can be found using the rate of submissions, score improvements and other statistical features from just their submission history. This is the origin of my idea.
## Necessity:
Summer of ML is a great initiative taken by our seniors to teach us ML. The regression based contest is our **first time participating in a kaggle contest**. Its a chance for us to show what we learnt in the summer. But, the AI csv submitters ruined that first contest. Doing submissions via AI in contest, and pushing back all the hard coders , hits the morale of the beginners. Leading to quitting the program, or making them cheat in the same way using AI in the next contests. This is a huge problem. There should be a way to flag who is cheating without actually the participants recognising. This is the main aim of my project.

---

## Tech Stack Used in the project

| Component | Tech |
|----------|------|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Model Training | scikit-learn, XGBClassifier, DecisionTreeClassifier |
| Frontend | Python |
| API | Fast API |
| Model Persistence | Pickle, Joblib |
| Deployment | streamlit |

---

## Machine Learning Models Used

- **Algorithm**: `XGBClassifier, DecisionTreeClassifier(for visualization purpose)`
- **Tuned via**: `Optuna + stratifiedKFold + cross validation`
- **Accuracy**: `0.86`
- **macro f1 Score**: `0.819`
- **visualization**: `dtreeviz + graphviz`

## Workflow:



