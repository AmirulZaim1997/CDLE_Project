package ai;


import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.medianBlur;

public class imageBlurring {
    public static void main(String[] args) throws IOException {
        // Load image
        Mat src = imread(new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath());
        Display.display(src, "Input");

        // Apply Gaussian blurring
        Mat dest_gauss = new Mat();
        GaussianBlur(src, dest_gauss, new Size(3, 3), 2);
        Display.display(dest_gauss, "Gaussian Blur");

        // Apply median blurring
        Mat dest_median = new Mat();
        medianBlur(src, dest_median, 3);
        Display.display(dest_median, "Median Blur");
    }
}
