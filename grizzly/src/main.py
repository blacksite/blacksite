import deepneuralnetwork as dnn

if __name__ == "__main__":

    option = input(
        '0: Load Deep Neural Network\n'
        '1: Train New Deep Neural Network\n'
    )

    while True:
        if option == 0:
            while True:
                filename = input('Enter the Deep Neural Network filename\n')
                dnn.load_dnn(filename)
            break
        elif option == 1:
            dnn.train_dnn()
            break
        else:
            print('Invalid input')
            option = input(
                '0: Load Deep Neural Network\n'
                '1: Train New Deep Neural Network\n'
            )
    # Start the callback to watch the dataset table
    # When a threshold is reached, a new DNN is trained
    # It replaces the current DNN
    dnn_retraining_thread = dnn.RetrainDNNThread(4, "RetrainDNNThread")
    dnn_retraining_thread.start()
    initial_detector_training_thread = dnn.TrainInitialDetectorsThread(1, "TrainInitialDetectorsThread")
    initial_detector_training_thread.start()
    detector_retraining_thread = dnn.RetrainDetectorsThread(2, "RetrainDetectorsThread")
    detector_retraining_thread.start()
    evaluate_initial_suspicious_instances_thread = dnn.EvaluateInitialSuspiciousInstanceThread(3, "EvaluateInitialSuspiciousInstanceThread")
    evaluate_initial_suspicious_instances_thread.start()
    evaluate_suspicious_instances_thread = dnn.EvaluateSuspiciousInstanceThread(4, "EvaluateSuspiciousInstanceThread")
    evaluate_suspicious_instances_thread.start()

    initial_detector_training_thread.join()
    evaluate_initial_suspicious_instances_thread.join()
    evaluate_suspicious_instances_thread.join()
    dnn_training_thread.join()
    detector_training_thread.join()

