using Microsoft.ML;
using SentimentAnalysisConsoleApp.DataStructures;
using System;
using static Microsoft.ML.DataOperationsCatalog;

namespace MachineLearningSA
{
    class Program
    {
        static void Main(string[] args)
        {

            //string DataPath = @"C:\\Users\\MSI PC\\source\\repos\\MachineLearingSA\\MachineLearingSA\\Datasets\\wikiDetoxAnnotated40kRows.tsv";
            string DataPath = @"C:\\Users\\MSI PC\\source\\repos\\MachineLearingSA\\MachineLearingSA\\Datasets\\fox-news-comments.tsv";
            string ModelPath = @"C:\\Users\\MSI PC\\source\\repos\\MachineLearingSA\\MachineLearingSA\\SentimentModel.zip";
            

            DataViewSchema modelSchema;

            var mlContext = new MLContext(seed: 1);

            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
            IDataView trainingData = trainTestSplit.TrainSet;
            IDataView testData = trainTestSplit.TestSet;

             
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentIssue.Text));

                               
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            //Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            //Save / persist the trained model to a.ZIP file
            mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);

            ////Load trained model
            //ITransformer trainedModel = mlContext.Model.Load(ModelPath, out modelSchema);

            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"The accuracy of this model : {metrics.Accuracy}\n");
   

          

            // TRY IT: Make a single test prediction, loading the model from .ZIP file
            SentimentIssue sampleStatement = new SentimentIssue { Text = "so you admit being a woman?" };     
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

  
            var resultprediction = predEngine.Predict(sampleStatement);
            Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction.Probability} ");
        }
    }
}
