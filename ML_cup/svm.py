import sklearn.svm as svm


def svm_predict(train_loader, test_loader, kernel='linear'):
    # Instantiate the SVM classifier
    svm_classifier = svm.SVC(kernel=kernel)  # You can use different kernels like 'linear', 'rbf', 'poly', etc.

    for batch_idx, (data, target) in enumerate(train_loader):
        # Train the SVM model
        svm_classifier.fit(data, target)

    for batch_idx, (data, target) in enumerate(test_loader):
        # Predict labels for the test set
        predictions = svm_classifier.predict(data)

        # Evaluate the model
        accuracy = svm_classifier.score(data, target)
        print(f"Accuracy: {accuracy}, index: {batch_idx}")
