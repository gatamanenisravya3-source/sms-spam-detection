# Sample Output

Actual results from running `python src/train.py` in a clean sandbox, using the real UCI SMS Spam Collection dataset (5,574 messages).

```
[OK] Dataset already exists: data/SMSSpamCollection
[INFO] Training model...
[INFO] Evaluating...

Confusion Matrix:
[[957   9]
 [ 10 139]]

Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99       966
        spam       0.94      0.93      0.94       149

    accuracy                           0.98      1115
   macro avg       0.96      0.96      0.96      1115
weighted avg       0.98      0.98      0.98      1115


[OK] Saved model to: models/sms_spam_model.joblib
```

## Live prediction examples (`src/predict.py`)

```
Message: 'Congratulations! You have won a free prize. Claim now!'
Prediction: SPAM (spam probability: 0.979)

Message: 'Hey, are we still meeting for lunch tomorrow?'
Prediction: HAM (spam probability: 0.086)
```

These numbers match the ~98% accuracy quoted in the main README and were produced by an actual training run, not hand-written.
