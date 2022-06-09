Tuple defaultBalance = Tuple(40, 60);
int defaultSampleSize = 1000;
List classifiers = ["LMNN", "SVM", "GMLVQ", "NaiveBayes", "LogReg"];
List estimators = ["CV", "RCV", "B632", "B632+"]
                        
/* Used to modify balance, or, keep same balance with default balance tuple 
and change sample size */
Function(List) modifyBalance(List dataFrame, Tuple balance, int sampleSize) {
    randomize(dataFrame)
    int firstClass = 0;
    int secondClass = 0;
    List newDataFrame = [];
    bool firstClassFlag = 1;
    bool secondClassFlag = 1;
    int firstClassCount = 0;
    int secondClassCount = 0;
    while (newDataFrame.size() != sampleSize) {
        if (firstClassFlag == 1 && dataFrame[i].target == 0) {
            newDataFrame.append(dataFrame[i]);
            firstClassCount++;
            if (firstClassCount == balance[0]) firstClassFlag = 0;
        } if (secondClassFlag == 1 && dataFrame[i].target == 1) {
            newDataFrame.append(dataFrame[i]);
            secondClassCount++;
            if (secondClassCount == balance[1]) secondClassFlag = 0;
        }
    }
    return newDataFrame;
}

Function(List) PCA(List dataFrame, int principalComponents) {
    // Using Library Functions
    dataFrame = Lib.PCA(dataFrame, principalComponents);
    return dataFrame;
    // Centre dataset by substracting the mean
} 

Function(void) performCrossValidation(string cf, Model model, string est, string sim)  {
    if (model is appropriate as cf) Lib.CrossValidate(10, model);
    else MyCrossValidation(10, myModel[?]);
}

Function(void) performRepeatedCV(string cf, Model model, string est, string sim) {
    int error = average(
        for (i in range(0:5)) {
            performCrossValidation(cf, model);
        }
    )
    errors[sim][cf][est].append(error);
}

Function(void) performBootstrap632(string cf, Model model, string est, string sim) {
    if (model is appropriate as cf) Lib.Bootstrap632(50, model);
    else MyBootstrap632(50, myModel[?]);
}

Function(void) performBootrstrap632plus(string cf, Model model, string est, string sim) {
    if (model is appropriate as cf) Lib.Bootstrap632plus(50, model);
    else MyBootstrap632plus(50, myModel[?]);
}

Function(void) train(List estimators, Model model, string cf, string sim) {
    for (est in estimators) {
        switch(est) {
            case "CV":
                performCrossValidation(cf, model, est, sim);
            case "RCV":
                performRepeatedCV(cf, model, est, sim);
            case "B632":
                performBootstrap632(cf, model, est, sim);
            case "B632+":
                performBootrstrap632plus(cf, model, est, sim);
        }
    }
}

Function(void) performEstimations(string cf, List dataFrameSS, string sim) {
    switch(cf) {
        case "LMNN":
            Model modelLMNN = Lib.LMNN(hyperparameters, dataFrameSS);
            train(estimators, modelLMNN, cf, sim);
        case "SVM":
            Model modelSVM = Lib.SVM(hyperparameters, dataFrameSS);
            train(estimators, modelSVM, cf, sim);
        case "GMLVQ":
            Model modelGMLVQ = Lib.GMLVQ(hyperparameters, dataFrameSS);
            train(estimators, modelGMLVQ, cf, sim);
        case "NaiveBayes":
            ModelNB = Lib.NaiveBayes(hyperparameters, dataFrameSS);
            train(estimators, ModelNB, cf, sim);
        case "LogReg":
            ModelLR = Lib.LogisticRegretion(hyperparameters, dataFrameSS);
            train(estimators, ModelLR, cf, sim);
    }
}

Function(void) simulationA(List sampleSizes, List dataFrame, string sim) {
    for (ss in sampleSizes) {
        List dataFrameSS = modifyBalance(dataFrame, defaultBalance, ss);
        for (cf in classifiers) {
            performEstimations(cf, dataFrameSS, sim);
        }
    }
}

Function(void) simulationB(List balances, List dataFrame, string sim) {
    for (b in balances) {
        List dataFrameB = modifyBalance(dataFrame, b, defaultSampleSize);
        for (cf in classifiers) {
            performEstimations(cf, dataFrameB, sim);
        }
    }
}

Function(void) simulationC(List principalComponents, List dataFrame, string sim) {
    for (pc in principalComponents) {
        List dataFrameTemp = modifyBalance(dataFrame, defaultBalance, defaultSampleSize)
        List dataFramePC = PCA(dataFrameTemp, pc);
        for (cf in classifiers) {
            performEstimations(cf, dataFramePC, sim);
        }
    }
}

Function(int) main() {
    List dataFrame = read("dfName.csv");
    List sampleSizes = [50,75,100,125,150,200,300,400,500,1000,1500,2000,3000,4000,4599]; // Percentages Instead
    List balances = [(30,70), (35,65), (40,60), (45, 65), (50,50)]; // Keep Full Sample Number4601 and do balance
    // List principalComponents = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58];
    // Amount of variance explained by PCA, analysis for dimentionality's role in training methodology

    for (i in range(0:50)) {
        simulationA(sampleSizes, dataFrame, "Simulation A");
        
        simulationB(balances, dataFrame, "Simulation B");
        
        // simulationC(principalComponents, dataFrame, "Simulation C");

        // Sofie's data PCA sorted 
    }
}

Dictionary errors = {"Simulation A": {
                        "LMNN": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "SVM": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "GMLVQ": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "NaiveBayes": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "LogReg": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } } "Simulation B": {
                        "LMNN": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "SVM": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "GMLVQ": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "NaiveBayes": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "LogReg": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } } "Simulation C": {
                        "LMNN": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "SVM": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "GMLVQ": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "NaiveBayes": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } "LogReg": {
                            "CV": [],
                            "RCV": [],
                            "B632": [],
                            "B632+": []
                        } }
                    }