package com.example.detection.classification;

/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import static com.example.detection.classification.Utils.average;
import static com.example.detection.classification.Utils.l2Norm2D;
import static com.example.detection.classification.Utils.multiplyAll;
import static com.example.detection.classification.Utils.subtract;
import static com.example.detection.classification.Utils.subtractAll;

import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.pose.PoseLandmark;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates embedding for given list of Pose landmarks.
 */
public class PoseEmbedding {
    // Multiplier to apply to the torso to get minimal body size. Picked this by experimentation.
    private static final float TORSO_MULTIPLIER = 2.5f;

    public static List<PointF3D> getPoseEmbedding(List<PointF3D> landmarks) {
        return normalize(landmarks);
    }

    private static List<PointF3D> normalize(List<PointF3D> landmarks) {
        List<PointF3D> normalizedLandmarks = new ArrayList<>(landmarks);
        // Normalize translation.
        PointF3D center = average(
                landmarks.get(PoseLandmark.LEFT_HIP), landmarks.get(PoseLandmark.RIGHT_HIP));
        subtractAll(center, normalizedLandmarks);

        // Normalize scale.
        multiplyAll(normalizedLandmarks, 1 / getPoseSize(normalizedLandmarks));
        return normalizedLandmarks;
    }

    // Translation normalization should've been done prior to calling this method.
    private static float getPoseSize(List<PointF3D> landmarks) {
        // Note: This approach uses only 2D landmarks to compute pose size as using Z wasn't helpful
        // in our experimentation but you're welcome to tweak.
        PointF3D hipsCenter = average(
                landmarks.get(PoseLandmark.LEFT_HIP), landmarks.get(PoseLandmark.RIGHT_HIP));

        PointF3D shouldersCenter = average(
                landmarks.get(PoseLandmark.LEFT_SHOULDER),
                landmarks.get(PoseLandmark.RIGHT_SHOULDER));

        float torsoSize = l2Norm2D(subtract(hipsCenter, shouldersCenter));
        float maxDistance = torsoSize * TORSO_MULTIPLIER;
        // torsoSize * TORSO_MULTIPLIER is the floor we want based on experimentation but actual size
        // can be bigger for a given pose depending on extension of limbs etc so we calculate that.
        for (PointF3D landmark : landmarks) {
            float distance = l2Norm2D(subtract(hipsCenter, landmark));
            if (distance > maxDistance) {
                maxDistance = distance;
            }
        }

        return maxDistance;
    }
    private PoseEmbedding() {}
}
