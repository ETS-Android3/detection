package com.example.detection.classification;


import static com.example.detection.classification.PoseEmbedding.getPoseEmbedding;
import android.content.Context;
import android.os.Trace;

import com.example.detection.ml.FinalModel1024;
import com.example.detection.ml.FinalModel2048;
import com.example.detection.ml.FinalModel3;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class DeepLearningClassification {
    private FinalModel3 model;
    private int countInference;
    private int totalMilli;

    public DeepLearningClassification(Context context, int countInference, int totalMilli) throws IOException {
        this.countInference = countInference;
        this.totalMilli = totalMilli;
        this.model = FinalModel3.newInstance(context);
        System.out.println("final model 2048");
    }

    public List<Integer> classify(Pose pose) {
        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 99}, DataType.FLOAT32);
        List<PointF3D> landmarks = extractPoseLandmarks(pose);
        if (landmarks.isEmpty()) {
            System.out.println("landmarks empty");
            return null;
        }
        List<PointF3D> embedding = getPoseEmbedding(landmarks);

        float temp[] = new float[99];
        int q = 0;
        for (PointF3D landmark : embedding) {
            temp[q] = (landmark.getX());
            temp[q+1] = (landmark.getY());
            temp[q+2] = (landmark.getZ());
            q=q+3;
        }
        inputFeature0.loadArray(temp,new int[]{1,99});

//        System.out.println("input: " + Arrays.toString(inputFeature0.getFloatArray()));

        long startTime = System.nanoTime();
        // Runs model inference and gets result.
        Trace.beginSection("DeepLearningClassification.classify");
        FinalModel3.Outputs outputs = model.process(inputFeature0);
        Trace.endSection();
        long endTime = System.nanoTime();
        int duration = (int) (endTime - startTime);
        countInference++;
        if(countInference<100){
            totalMilli = totalMilli + duration;
        }
        if(countInference == 100){
            System.out.println("Average inference time pose classification: " + totalMilli/100);
        }
        List<Integer> list = new ArrayList<>();
        list.add(countInference);
        list.add(totalMilli);

        System.out.println(countInference + " execute pose classification, nano seconds: " + duration);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        float[] data = outputFeature0.getFloatArray();
//        System.out.println("output1: " + Arrays.toString(data));
        int maxAt = 0;
        for (int j = 0; j < data.length; j++) {
            maxAt = data[j] > data[maxAt] ? j : maxAt;
        }

        switch (maxAt) {
            case 0:
                System.out.println("Goddess, confidence:" + data[maxAt]);
                break;
            case 1:
                System.out.println("Warrior 2, confidence:" + data[maxAt]);
                break;
            case 2:
                System.out.println("Tree, confidence:" + data[maxAt]);
                break;
            case 3:
                System.out.println("Plank, confidence:" + data[maxAt]);
                break;
            case 4:
                System.out.println("Downward Facing Dog, confidence:" + data[maxAt]);
                break;
        }
        // Releases model resources if no longer used.
        model.close();
        return list;
    }


    private static List<PointF3D> extractPoseLandmarks(Pose pose) {
        List<PointF3D> landmarks = new ArrayList<>();
        for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
            landmarks.add(poseLandmark.getPosition3D());
        }
        return landmarks;
    }


}
