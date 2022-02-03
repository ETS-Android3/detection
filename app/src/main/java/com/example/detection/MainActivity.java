package com.example.detection;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Trace;
import android.util.Log;
import android.util.Size;
import android.view.Surface;


import androidx.annotation.GuardedBy;
import androidx.annotation.NonNull;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.app.Fragment;

import com.example.detection.classification.DeepLearningClassification;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener {
    private static final String TAG = "DeeplearningClassification";
    private int sensorOrientation;
    private PoseDetector poseDetector;
    private boolean firstTime;
    private Context context;
    private int countInference;
    private long totalMilli;
    private int countClassification;
    private int totalNano;
    private DeepLearningClassification deepLearningClassification;
    //    private GraphicOverlay graphicOverlay;
    // To keep the latest images and its metadata.
    @GuardedBy("this")
    private ByteBuffer latestImage;

    // To keep the images and metadata in process.
    @GuardedBy("this")
    private ByteBuffer processingImage;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        firstTime = true;
        context = this;
        setContentView(R.layout.activity_main);
        createPoseDetector();
        countInference = 0;
        totalMilli = 0;
        countClassification = 0;
        totalNano = 0;
//        graphicOverlay = findViewById(R.id.graphic_overlay);
//        if (graphicOverlay == null) {
//            Log.d(TAG, "graphicOverlay is null");
//        }

        //TODO ask for camera permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA}, 121);
        } else {
            //TODO show live camera footage
            setFragment();
        }
    }

    private void createPoseDetector() {
        // Accurate pose detector on static images, when depending on the pose-detection-accurate sdk
        PoseDetectorOptions options =
                new PoseDetectorOptions.Builder()
                        .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                        .build();
        poseDetector = PoseDetection.getClient(options);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        //TODO show live camera footage
        if (grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            setFragment();
        } else {
            finish();
        }
    }

    //TODO fragment which show live footage from camera
    int previewHeight = 0, previewWidth = 0;

    protected void setFragment() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        String cameraId = null;
        try {
            cameraId = manager.getCameraIdList()[0];
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Fragment fragment;
        CameraConnectionFragment camera2Fragment =
                CameraConnectionFragment.newInstance(
                        (size, rotation) -> {
                            previewHeight = size.getHeight();
                            previewWidth = size.getWidth();
                            Log.d("tryOrientation", "rotation: " + rotation + "   orientation: " + getScreenOrientation() + "  " + previewWidth + "   " + previewHeight);
                            sensorOrientation = rotation - getScreenOrientation();
                        },
                        this,
                        R.layout.camera_fragment,
                        new Size(640, 480));

        camera2Fragment.setCamera(cameraId);
        fragment = camera2Fragment;
        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }


    //TODO getting frames of live camera footage and passing them to model
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap;

    @Override
    public void onImageAvailable(ImageReader reader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

//            System.out.println("is processing frame true?");
            if (isProcessingFrame) {
//                System.out.println("processing frame is true");
                image.close();
                return;
            }
//            System.out.println("yes hello i'm gonna process!!");
            isProcessingFrame = true;
            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter =
                    new Runnable() {
                        @Override
                        public void run() {
                            ImageUtils.convertYUV420ToARGB8888(
                                    yuvBytes[0],
                                    yuvBytes[1],
                                    yuvBytes[2],
                                    previewWidth,
                                    previewHeight,
                                    yRowStride,
                                    uvRowStride,
                                    uvPixelStride,
                                    rgbBytes);
                        }
                    };

            postInferenceCallback =
                    new Runnable() {
                        @Override
                        public void run() {
                            image.close();
                            isProcessingFrame = false;
                        }
                    };
            processImage();

        } catch (final Exception e) {

            return;
        }

    }

    private void runPoseDetection(Bitmap bitmap) {
        InputImage image = InputImage.fromBitmap(bitmap, sensorOrientation);
        Trace.beginSection("MainActivity.runPoseDetection");
        long startTime = System.nanoTime();
        Task<Pose> result =
                poseDetector.process(image)
                        .addOnSuccessListener(
                                new OnSuccessListener<Pose>() {
                                    @Override
                                    public void onSuccess(Pose pose) {
                                        Trace.endSection();
                                        long endTime = System.nanoTime();
                                        long duration = (endTime - startTime);
                                        countInference++;
                                        if (countInference < 100) {
                                            totalMilli = totalMilli + duration;
                                        }
                                        if (countInference == 100) {
                                            System.out.println("Average inference time pose detection: " + totalMilli / 100);
                                        }
                                        System.out.println(countInference + " execute pose detection milli seconds: " + duration);
                                        if (!pose.getAllPoseLandmarks().isEmpty()) {
                                            System.out.println("pose detected");
                                            try {
                                                deepLearningClassification = new DeepLearningClassification(context, countClassification, totalNano);
                                            } catch (IOException e) {
                                                e.printStackTrace();
                                            }

                                        }


                                    }
                                })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        Trace.endSection();
                                        System.out.println("pose detection failed, oops");
                                    }
                                });
    }


    private void processImage() {
//        System.out.println("process image");
        imageConverter.run();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        runPoseDetection(rgbFrameBitmap);
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        postInferenceCallback.run();
    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

}
