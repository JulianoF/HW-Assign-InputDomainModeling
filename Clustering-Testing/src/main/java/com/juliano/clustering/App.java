package com.juliano.clustering;

import java.io.File;

import net.sf.javaml.clustering.evaluation.AICScore;
import net.sf.javaml.clustering.evaluation.BICScore;
import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;

import net.sf.javaml.clustering.AQBC;
import net.sf.javaml.clustering.SOM;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

public class App {
    public static void main(String[] args) throws Exception {
        /* Load a dataset */
        Dataset data = FileHandler.loadDataset(new File("datasets/iris.data"), 4, ",");
        /*
         * Create a new instance of the KMeans algorithm, with no options
         * specified. By default this will generate 4 clusters.
         */
        Clusterer km = new KMeans();
        Clusterer adapt = new AQBC();
        Clusterer somTest = new SOM();
        /*
         * Cluster the data, it will be returned as an array of data sets, with
         * each dataset representing a cluster
         */
        Dataset[] clusters1 = km.cluster(data);
        Dataset[] clusters2 = adapt.cluster(data);
        Dataset[] clusters3 = somTest.cluster(data);

        System.out.println("\n******************************************************");
        System.out.println("KMEAN Cluster count: " + clusters1.length);
        System.out.println("AQBC Cluster count: " + clusters2.length);
        System.out.println("SOM Cluster count: " + clusters3.length);
        System.out.println("******************************************************\n");

        Clusterer km3 = new KMeans(3);
        Clusterer km4 = new KMeans(4);
        /*
         * Cluster the data, we will create 3 and 4 clusters.
         */
        Dataset[] clusters4 = km3.cluster(data);
        Dataset[] clusters5 = km4.cluster(data);

        ClusterEvaluation aic = new AICScore();
        ClusterEvaluation bic = new BICScore();
        ClusterEvaluation sse = new SumOfSquaredErrors();

        double aicScore3 = aic.score(clusters4);
        double bicScore3 = bic.score(clusters4);
        double sseScore3 = sse.score(clusters4);

        double aicScore4 = aic.score(clusters5);
        double bicScore4 = bic.score(clusters5);
        double sseScore4 = sse.score(clusters5);

        System.out.println("AIC score: " + aicScore3 + "\t" + aicScore4);
        System.out.println("BIC score: " + bicScore3 + "\t" + bicScore4);
        System.out.println("Sum of squared errors: " + sseScore3 + "\t" + sseScore4);
    }
}
