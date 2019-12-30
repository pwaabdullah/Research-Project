package org.deeplearning4j.examples.dataexamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * @author Adam Gibson
 */
public class CSV_DoA_ver_2 {

    private static Logger log = LoggerFactory.getLogger(CSV_DoA_ver_2.class);

    public static void main(String[] args) throws  Exception {

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';

        final String filenameTrain  = new ClassPathResource("train_P7.txt").getFile().getPath();
        final String filenameTest  = new ClassPathResource("train_P_1_2_6_19.txt").getFile().getPath();

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 128;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 2;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize1 = 580; //470;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        int batchSize2 = 1500; //70; 58; //140 699;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        //Load the training data:

        RecordReader TrainRecord = new CSVRecordReader();
        TrainRecord.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(TrainRecord,batchSize1,labelIndex,numClasses);
        DataSet TrainData = trainIter.next();


        //Load the test/evaluation data:

        RecordReader TestRecord = new CSVRecordReader();
        TestRecord.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(TestRecord,batchSize2,labelIndex,numClasses);
        DataSet TestData = testIter.next();


        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(TrainData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(TrainData);     //Apply normalization to the training data
        normalizer.transform(TestData);      //Apply normalization to the test data. This is using statistics calculated from the *training* set


        final int numInputs = 128;
        int outputNum = 2;
        int iterations = 1000;
        long seed = 6;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.1)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(100)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(100).nOut(100)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(100).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(TrainData);

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(2);
        INDArray output = model.output(TestData.getFeatureMatrix());
        eval.eval(TestData.getLabels(), output);
        log.info(eval.stats());
    }

}

