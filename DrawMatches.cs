
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.UI;
using Emgu.CV.GPU;

namespace SURFFeature2
{
    public static class DrawMatches
    {
        public static Image<Bgr, Byte> Draw(String modelImageFileName, String observedImageFileName, out long matchTime)
        {
            Image<Gray, Byte> modelImage = new Image<Gray, byte>(modelImageFileName);
            Image<Gray, Byte> observedImage = new Image<Gray, byte>(observedImageFileName);
            Stopwatch watch;
            HomographyMatrix homography = null;

            SURFDetector surfCPU = new SURFDetector(600, false);
            
            //SIFTDetector surfCPU = new SIFTDetector();
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            Matrix<int> indices;

            Matrix<byte> mask;
            int k = 2;
            double uniquenessThreshold = 0.8;
       
                //extract features from the object image
                modelKeyPoints = surfCPU.DetectKeyPointsRaw(modelImage, null);
                Matrix<float> modelDescriptors = surfCPU.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

                watch = Stopwatch.StartNew();

                // extract features from the observed image
                observedKeyPoints = surfCPU.DetectKeyPointsRaw(observedImage, null);
                Matrix<float> observedDescriptors = surfCPU.ComputeDescriptorsRaw(observedImage, null, observedKeyPoints);
                BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(DistanceType.L2);
                matcher.Add(modelDescriptors);

                indices = new Matrix<int>(observedDescriptors.Rows, k);
                using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, k))
                {
                    matcher.KnnMatch(observedDescriptors, indices, dist, k, null);
                    mask = new Matrix<byte>(dist.Rows, 1);
                    mask.SetValue(255);
                    Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
                }

                int nonZeroCount = CvInvoke.cvCountNonZero(mask);
                if (nonZeroCount >= 4)
                {
                    //nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                    if (nonZeroCount >= 4)
                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
                }

                watch.Stop();

            Image<Bgr, Byte> modelImage2 = new Image<Bgr, Byte>(modelImageFileName);
            Image<Bgr, Byte> observedImage2 = new Image<Bgr, Byte>(observedImageFileName);

            Image<Bgr, Byte> result = Features2DToolbox.DrawMatches(modelImage2, modelKeyPoints, observedImage2, observedKeyPoints,
               indices, new Bgr(255, 255, 255), new Bgr(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.DEFAULT);

            if (homography != null)
            { 
                Rectangle rect = modelImage.ROI;
                PointF[] pts = new PointF[] { 
               new PointF(rect.Left, rect.Bottom),
               new PointF(rect.Right, rect.Bottom),
               new PointF(rect.Right, rect.Top),
               new PointF(rect.Left, rect.Top)};
                homography.ProjectPoints(pts);

                result.DrawPolyline(Array.ConvertAll<PointF, Point>(pts, Point.Round), true, new Bgr(Color.Red), 5);
            }
            for (int i = 0; i < observedKeyPoints.Size; ++i)
            {
                Color color = Color.FromArgb((int)observedImage2[(int)observedKeyPoints[i].Point.Y, (int)observedKeyPoints[i].Point.X].Red, (int)observedImage2[(int)observedKeyPoints[i].Point.Y, (int)observedKeyPoints[i].Point.X].Green, (int)observedImage2[(int)observedKeyPoints[i].Point.Y, (int)observedKeyPoints[i].Point.X].Blue);
                float hue = color.GetHue();
                float sat = color.GetSaturation();
                float bright = color.GetBrightness();
                float satThr = (float) 0.0f / 240.0f;
                float brightTrh = (float) 40.0f / 240.0f;
                float brightThr2 = (float) 15.0f / 24.0f;
                if (sat < satThr && bright < brightTrh)
                    continue;
                if (bright > brightThr2)
                {
                    result.Draw(new CircleF(observedKeyPoints[i].Point, 4), new Bgr(Color.White), -1);
                    continue;
                }
                if(hue>230)//rosu
                    result.Draw(new CircleF(observedKeyPoints[i].Point, 4), new Bgr(Color.Red), -1);
                //else if(hue>180)//mov
                //    result.Draw(new CircleF(observedKeyPoints[i].Point, 4), new Bgr(Color.Purple), -1);
                else if(hue>120)//albastru
                    result.Draw(new CircleF(observedKeyPoints[i].Point, 4), new Bgr(Color.Blue), -1);
                else if (hue>60) //verde               
                    result.Draw(new CircleF(observedKeyPoints[i].Point, 4), new Bgr(Color.Yellow), -1);               
                else if (hue>30)//galben               
                    result.Draw(new CircleF(observedKeyPoints[i].Point, 4), new Bgr(Color.Yellow), -1);
                else result.Draw(new CircleF(observedKeyPoints[i].Point, 4), new Bgr(Color.Red), -1);
                
            }
 
            matchTime = watch.ElapsedMilliseconds;

            return result; 
        }

        public static Image<Bgr, Byte> DrawLines(String modelImageFileName, String observedImageFileName, out long matchTime)
        {
            //Image<Gray, Byte> cannyEdges = gray.Canny(cannyThreshold, cannyThresholdLinking);
            //Load the image from file
            Image<Bgr, Byte> observedImage = new Image<Bgr, byte>(observedImageFileName);
            Stopwatch watch;
            watch = Stopwatch.StartNew();

            Image<Gray, Byte> graySoft = observedImage.Convert<Gray, Byte>();//.PyrDown().PyrUp();
            //ImageViewer.Show(graySoft, "graysoft"); 
            //Image<Gray, Byte> gray = graySoft.SmoothGaussian(3);
            //ImageViewer.Show(gray, "graysoft"); 
            //gray = gray.AddWeighted(graySoft, 1.5, -0.5, 0);
            //ImageViewer.Show(graySoft, "graysoft"); 

            Gray cannyThreshold = new Gray(80);
            Gray cannyThresholdLinking = new Gray(200);
            Gray circleAccumulatorThreshold = new Gray(1000);

            Image<Gray, Byte> cannyEdges = graySoft.Canny(cannyThreshold, cannyThresholdLinking);

            //Circles
            LineSegment2D[][] lines = cannyEdges.HoughLinesBinary(0.1, Math.PI / 45.0,1,20, 1.0);

            //draw circles (on original image)
            foreach (LineSegment2D[] line in lines)
                foreach (LineSegment2D l in line)
                    observedImage.Draw(l, new Bgr(Color.Red), 2);

            watch.Stop();
            matchTime = watch.ElapsedMilliseconds;
            return observedImage;
        }
        public static Image<Bgr, Byte> Parallelogram(String modelImageFileName, String observedImageFileName, out long matchTime)
        {
            //Image<Gray, Byte> cannyEdges = gray.Canny(cannyThreshold, cannyThresholdLinking);
            //Load the image from file
            Image<Bgr, Byte> observedImage = new Image<Bgr, byte>(observedImageFileName);
            Stopwatch watch;
            HomographyMatrix homography = null;
            watch = Stopwatch.StartNew();

            Image<Gray, Byte> graySoft = observedImage.Convert<Gray, Byte>();//.PyrDown().PyrUp();
            //ImageViewer.Show(graySoft, "graysoft"); 
            //Image<Gray, Byte> gray = graySoft.SmoothGaussian(3);
            //ImageViewer.Show(gray, "graysoft"); 
            //gray = gray.AddWeighted(graySoft, 1.5, -0.5, 0);
            //ImageViewer.Show(graySoft, "graysoft"); 

            Gray cannyThreshold = new Gray(149);
            Gray cannyThresholdLinking = new Gray(149);
            Gray circleAccumulatorThreshold = new Gray(1000);

            Image<Gray, Byte> cannyEdges = graySoft.Canny(cannyThreshold, cannyThresholdLinking);
            Image<Gray,Byte> modelImage = new Image<Gray,Byte>(modelImageFileName).Canny(cannyThreshold, cannyThresholdLinking); 
            SURFDetector surfCPU = new SURFDetector(200, false);
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            Matrix<int> indices;

            Matrix<byte> mask;
            int k = 2;
            double uniquenessThreshold = 0.99;

            //extract features from the object image
            modelKeyPoints = surfCPU.DetectKeyPointsRaw(modelImage, null);
            Matrix<float> modelDescriptors = surfCPU.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

            watch = Stopwatch.StartNew();

            // extract features from the observed image
            observedKeyPoints = surfCPU.DetectKeyPointsRaw(cannyEdges, null);
            Matrix<float> observedDescriptors = surfCPU.ComputeDescriptorsRaw(cannyEdges, null, observedKeyPoints);
            BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(DistanceType.L2);
            matcher.Add(modelDescriptors);

            indices = new Matrix<int>(observedDescriptors.Rows, k);
            using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, k))
            {
                matcher.KnnMatch(observedDescriptors, indices, dist, k, null);
                mask = new Matrix<byte>(dist.Rows, 1);
                mask.SetValue(255);
                Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
            }

            int nonZeroCount = CvInvoke.cvCountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                //nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
            }

            watch.Stop();

            //Image<Bgr, Byte> modelImage2 = new Image<Bgr, Byte>(modelImageFileName);
            //Image<Bgr, Byte> observedImage2 = new Image<Bgr, Byte>(observedImageFileName);

            Image<Bgr, Byte> result = Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, cannyEdges, observedKeyPoints,
               indices, new Bgr(255, 255, 255), new Bgr(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.DEFAULT);

            if (homography != null)
            {
                Rectangle rect = modelImage.ROI;
                PointF[] pts = new PointF[] { 
               new PointF(rect.Left, rect.Bottom),
               new PointF(rect.Right, rect.Bottom),
               new PointF(rect.Right, rect.Top),
               new PointF(rect.Left, rect.Top)};
                homography.ProjectPoints(pts);

                result.DrawPolyline(Array.ConvertAll<PointF, Point>(pts, Point.Round), true, new Bgr(Color.Red), 5);
            }
            watch.Stop();
            matchTime = watch.ElapsedMilliseconds;
            return result;
        }
    }
}
