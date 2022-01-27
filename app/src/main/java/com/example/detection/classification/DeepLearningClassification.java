package com.example.detection.classification;


import static com.example.detection.classification.PoseEmbedding.getPoseEmbedding;
import android.content.Context;
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

    public DeepLearningClassification(Context context) throws IOException {
        this.model = FinalModel3.newInstance(context);
    }

    public List<String> classify(Pose pose) {
        ClassificationResult result = new ClassificationResult();
        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 99}, DataType.FLOAT32);
        List<PointF3D> landmarks = extractPoseLandmarks(pose);
        if (landmarks.isEmpty()) {
            System.out.println("landmarks empty");
            return null;
        }
        List<PointF3D> embedding = getPoseEmbedding(landmarks);
        ByteBuffer buf = ByteBuffer.allocateDirect(99 * 4);

        for (PointF3D landmark : embedding) {
            buf.putFloat(landmark.getX());
            buf.putFloat(landmark.getY());
            buf.putFloat(landmark.getZ());
        }
        buf.rewind();
        inputFeature0.loadBuffer(buf);

        System.out.println("input: " + Arrays.toString(inputFeature0.getFloatArray()));

        // Runs model inference and gets result.
        FinalModel3.Outputs outputs = model.process(inputFeature0);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        float[] data = outputFeature0.getFloatArray();
        System.out.println("output1: " + Arrays.toString(data));
        int i = 0;
        for (float val : data) {
            switch (i) {
                case 0:
                    result.putClassConfidence("goddess", val);
                    break;
                case 1:
                    result.putClassConfidence("warrior 2", val);
                    break;
                case 2:
                    result.putClassConfidence("tree", val);
                    break;
                case 3:
                    result.putClassConfidence("plank", val);
                    break;
                case 4:
                    result.putClassConfidence("downward facing dog", val);
                    break;
            }
            i++;
        }
        // Releases model resources if no longer used.
        model.close();
        return getStringFromResult(result);
    }

    public void classifySample(PoseSample sample) {
        boolean justOnce = true;
        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 99}, DataType.FLOAT32);
        List<PointF3D> landmarks = sample.getLandmarks();
        List<PointF3D> embedding = getPoseEmbedding(landmarks);
        ByteBuffer buf = ByteBuffer.allocateDirect(99 * 4);
        float temp[] = new float[99];
        int q = 0;
        for (PointF3D landmark : embedding) {
            temp[q] = (landmark.getX());
            temp[q+1] = (landmark.getY());
            temp[q+2] = (landmark.getZ());

            buf.putFloat(landmark.getX());
            buf.putFloat(landmark.getY());
            buf.putFloat(landmark.getZ());
            q=q+3;
        }
        buf.rewind();
        inputFeature0.loadBuffer(buf);

//            float y = 1;
//            for (int i = 0; i < 99; i++) {
//                y = buf.getFloat();
//                if (temp[i] == y) {
//                    System.out.println("same same! " + y);
//                } else {
//                    System.out.println("different! " + temp + " " + y);
//                }
//            }
//            buf.rewind();
        System.out.println("input2: " + Arrays.toString(inputFeature0.getFloatArray()));

        // Runs model inference and gets result.
        FinalModel3.Outputs outputs = model.process(inputFeature0);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        float[] data = outputFeature0.getFloatArray();
        System.out.println("output2: " + Arrays.toString(data));
        int maxAt = 0;
        for (int j = 0; j < data.length; j++) {
            maxAt = data[j] > data[maxAt] ? j : maxAt;
        }
        System.out.println("should be: " + sample.getClassName());
        System.out.println("found: " + maxAt);

        // Releases model resources if no longer used.
        model.close();
    }


    private static List<PointF3D> extractPoseLandmarks(Pose pose) {
        List<PointF3D> landmarks = new ArrayList<>();
        for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
            landmarks.add(poseLandmark.getPosition3D());
        }
        return landmarks;
    }

    private List<String> getStringFromResult(ClassificationResult classificationResult){
        List<String> listFromResult = new ArrayList<>();
        String conf;
        Set<String> classes = classificationResult.getAllClasses();
        for(String cl : classes) {
            float res = classificationResult.getClassConfidence(cl);
            conf = String.format(
                    Locale.US, "%s : %.2f confidence", cl, res);
            listFromResult.add(conf);
        }
        return listFromResult;
    }
}
