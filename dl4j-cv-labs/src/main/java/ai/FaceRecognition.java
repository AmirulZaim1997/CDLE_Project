package ai;

import ai.detection.FaceDetector;
import ai.detection.FaceLocalization;
import ai.detection.OpenCV_DeepLearningFaceDetector;
import ai.detection.OpenCV_HaarCascadeFaceDetector;
import ai.identification.DistanceFaceIdentifier;
import ai.identification.FaceIdentifier;
import ai.identification.Prediction;
import ai.identification.feature.InceptionResNetFeatureProvider;
import ai.identification.feature.VGG16FeatureProvider;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public class FaceRecognition {
    private static final Logger log = LoggerFactory.getLogger(FaceRecognition.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "Face Recognition ";



    public static void main(String[] args) throws IOException, ClassNotFoundException {
        //face identifier and face detector
        FaceDetector faceDetector = getFaceDetector(FaceDetector.OPENCV_DL_FACEDETECTOR);
        FaceIdentifier faceIdentifier = getFaceIdentifier(FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT);

        //stream video frame from camera
        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
        namedWindow(outputWindowsName, WINDOW_NORMAL);
        resizeWindow(outputWindowsName, 1280, 720);

        if (!capture.open(0)) {
            System.out.println("Cannot open the camera !!!");
        }

        Mat image = new Mat();
        Mat cloneCopy = new Mat();

        while (capture.read(image)) {
            flip(image, image, 1);

            //  Perform face detection
            image.copyTo(cloneCopy);
//            FaceDetector faceDetector1 = new FaceDetector();
//            faceDetector1.detectFaces(cloneCopy);
            faceDetector.detectFaces(cloneCopy);
            List<FaceLocalization> faceLocalizations = faceDetector.getFaceLocalization();
            annotateFaces(faceLocalizations, image);

            //Perform face recognition
            image.copyTo(cloneCopy);
            List<List<Prediction>> faceIdentities = faceIdentifier.recognize(faceLocalizations, cloneCopy);
            labelIndividual(faceIdentities, image);

            // Display output in a window
            imshow(outputWindowsName, image);

            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27) {
                destroyAllWindows();
                break;
            }
        }



    }

    private static void labelIndividual(List<List<Prediction>> faceIdentities, Mat image) {
        for (List<Prediction> i: faceIdentities){
            for(int j=0; j<i.size(); j++)
            {
                putText(
                        image,
                        i.get(j).toString()+ " (" + new DecimalFormat("0.00").format(i.get(j).getScore() * 100.00) + "%)",
                        new Point(
                                (int)i.get(j).getFaceLocalization().getLeft_x() + 2,
                                (int)i.get(j).getFaceLocalization().getLeft_y() - 5
                        ),
                        FONT_HERSHEY_COMPLEX,
                        0.5,
                        Scalar.YELLOW
                );
            }
        }
    }

    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {

        for (FaceLocalization i : faceLocalizations){
            rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(0, 255, 0, 0),2,8,0);
        }
    }

    private static FaceDetector getFaceDetector(String faceDetector) throws IOException {
        switch (faceDetector) {
            case FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR:
                return new OpenCV_HaarCascadeFaceDetector();
            case FaceDetector.OPENCV_DL_FACEDETECTOR:
                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
            default:
                return  null;
        }
    }

    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException {
        switch (faceIdentifier) {
            case FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT:
                return new DistanceFaceIdentifier(
                        new VGG16FeatureProvider(),
                        new ClassPathResource("FaceDB").getFile(), 0.5, 2);
            case FaceIdentifier.FEATURE_DISTANCE_INCEPTION_RESNET_PREBUILT:
                return new DistanceFaceIdentifier(
                        new InceptionResNetFeatureProvider(),
                        new ClassPathResource("FaceDB").getFile(), 0.3, 2);
            default:
                return null;
        }
    }


}
