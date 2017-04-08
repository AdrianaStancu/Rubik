using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SURFFeature2
{
    public class CameraCapture
    {
        private static Capture capture;
        private static bool retry;
        private static String[] colorConfig1;

        private static Image<Bgr, Byte> DetectHoughLines(Image<Gray, Byte> imageCanny, Image<Bgr, Byte> imageFrame)
        {
            LineSegment2D[][] lines = imageCanny.HoughLinesBinary(1, Math.PI / 45.0, 20, 30, 10);

            foreach (LineSegment2D[] line in lines)
                foreach (LineSegment2D l in line)
                    imageFrame.Draw(l, new Bgr(Color.Red), 2);
            return imageFrame;
        }
        private static List<KeyValuePair<int, Rectangle>> MatchPattern(List<Contour<Point>> rectangleContours, Image<Gray, Byte> imageCanny)
        {
            List<KeyValuePair<int, Rectangle>> rectangle = new List<KeyValuePair<int, Rectangle>>();
            ImageViewer.Show(imageCanny);

            List<KeyValuePair<float, Rectangle>> maxes = new List<KeyValuePair<float, Rectangle>>();
            foreach (Contour<Point> contour in rectangleContours)
            {
                Point[] vContour = contour.ToArray();
                for (int i = 0; i < vContour.Length; ++i)
                {
                    vContour[i].X = vContour[i].X - contour.BoundingRectangle.Location.X + (int)(contour.BoundingRectangle.Width * 0.1);
                    vContour[i].Y = vContour[i].Y - contour.BoundingRectangle.Location.Y + (int)(contour.BoundingRectangle.Height * 0.1);
                }

                int width = (int)(contour.BoundingRectangle.Width + contour.BoundingRectangle.Width * 0.3);
                int height = (int)(contour.BoundingRectangle.Height + contour.BoundingRectangle.Height * 0.3);

                Image<Gray, float> filtruPartial = new Image<Gray, float>(width, height, new Gray(0));
                filtruPartial.Draw(new LineSegment2D(new Point(vContour[0].X, vContour[0].Y),
                    new Point(vContour[1].X, vContour[1].Y)), new Gray(1), 2);
                filtruPartial.Draw(new LineSegment2D(new Point(vContour[1].X, vContour[1].Y),
                    new Point(vContour[2].X, vContour[2].Y)), new Gray(1), 2);
                filtruPartial.Draw(new LineSegment2D(new Point(vContour[2].X, vContour[2].Y),
                    new Point(vContour[3].X, vContour[3].Y)), new Gray(1), 2);
                filtruPartial.Draw(new LineSegment2D(new Point(vContour[3].X, vContour[3].Y),
                    new Point(vContour[0].X, vContour[0].Y)), new Gray(1), 2);
                filtruPartial = filtruPartial.SmoothBlur(6, 6);
                //ImageViewer.Show(filtruPartial);
                Image<Gray, float> filtruIntreg = new Image<Gray, float>(filtruPartial.Width * 3, filtruPartial.Height * 3, new Gray(0));
                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(0, 0, filtruPartial.Width, filtruPartial.Height)));
                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(0, filtruPartial.Height, filtruPartial.Width, filtruPartial.Height)));
                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(0, filtruPartial.Height * 2, filtruPartial.Width, filtruPartial.Height)));

                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(filtruPartial.Width, 0, filtruPartial.Width, filtruPartial.Height)));
                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(filtruPartial.Width, filtruPartial.Height, filtruPartial.Width, filtruPartial.Height)));
                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(filtruPartial.Width, filtruPartial.Height * 2, filtruPartial.Width, filtruPartial.Height)));

                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(filtruPartial.Width * 2, 0, filtruPartial.Width, filtruPartial.Height)));
                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(filtruPartial.Width * 2, filtruPartial.Height, filtruPartial.Width, filtruPartial.Height)));
                filtruPartial.CopyTo(filtruIntreg.GetSubRect(new Rectangle(filtruPartial.Width * 2, filtruPartial.Height * 2, filtruPartial.Width, filtruPartial.Height)));

                //ImageViewer.Show(filtruIntreg);

                int filterWidth = filtruIntreg.Width + contour.BoundingRectangle.Location.X <= imageCanny.Width ?
                    filtruIntreg.Width :
                    imageCanny.Width - contour.BoundingRectangle.Location.X;
                int filterHeight = filtruIntreg.Height + contour.BoundingRectangle.Location.Y <= imageCanny.Height ?
                    filtruIntreg.Height :
                    imageCanny.Height - contour.BoundingRectangle.Location.Y;
                Image<Gray, byte>[] filterImages = new Image<Gray, byte>[9];
                for (int i = 0; i < 9; ++i)
                {
                    filterImages[i] = new Image<Gray, byte>(filterWidth, filterHeight);
                }
                List<Rectangle> subRect = new List<Rectangle>();
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X,
                    contour.BoundingRectangle.Location.Y, filterWidth, filterHeight));
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X - filtruPartial.Width > 0 ?
                    contour.BoundingRectangle.Location.X - filtruPartial.Width : 0,
                    contour.BoundingRectangle.Location.Y,
                    filterWidth, filterHeight));
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X - (filtruPartial.Width * 2) > 0 ?
                    contour.BoundingRectangle.Location.X - (filtruPartial.Width * 2) : 0,
                    contour.BoundingRectangle.Location.Y,
                    filterWidth, filterHeight));
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X,
                    contour.BoundingRectangle.Location.Y - filtruPartial.Height > 0 ?
                    contour.BoundingRectangle.Location.Y - filtruPartial.Height : 0,
                    filterWidth, filterHeight));
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X - filtruPartial.Width > 0 ?
                    contour.BoundingRectangle.Location.X - filtruPartial.Width : 0,
                    contour.BoundingRectangle.Location.Y - filtruPartial.Height > 0 ?
                    contour.BoundingRectangle.Location.Y - filtruPartial.Height : 0,
                    filterWidth, filterHeight));
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X - (filtruPartial.Width * 2) > 0 ?
                    contour.BoundingRectangle.Location.X - (filtruPartial.Width * 2) : 0,
                    contour.BoundingRectangle.Location.Y - filtruPartial.Height > 0 ?
                    contour.BoundingRectangle.Location.Y - filtruPartial.Height : 0,
                    filterWidth, filterHeight));
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X,
                    contour.BoundingRectangle.Location.Y - (filtruPartial.Height * 2) > 0 ?
                    contour.BoundingRectangle.Location.Y - (filtruPartial.Height * 2) : 0,
                    filterWidth, filterHeight));
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X - filtruPartial.Width > 0 ?
                    contour.BoundingRectangle.Location.X - filtruPartial.Width : 0,
                    contour.BoundingRectangle.Location.Y - (filtruPartial.Height * 2) > 0 ?
                    contour.BoundingRectangle.Location.Y - (filtruPartial.Height * 2) : 0,
                    filterWidth, filterHeight));
                subRect.Add(new Rectangle(contour.BoundingRectangle.Location.X - (filtruPartial.Width * 2) > 0 ?
                    contour.BoundingRectangle.Location.X - (filtruPartial.Width * 2) : 0,
                    contour.BoundingRectangle.Location.Y - (filtruPartial.Height * 2) > 0 ?
                    contour.BoundingRectangle.Location.Y - (filtruPartial.Height * 2) : 0,
                    filterWidth, filterHeight));

                //ImageViewer.Show(imageCanny);
                for (int i = 0; i < 9; ++i)
                {
                    imageCanny.GetSubRect(subRect[i]).CopyTo(filterImages[i]);
                }

                //CvInvoke.cvFilter2D(imageCanny.GetSubRect(contour.BoundingRectangle), rezFilter, filtruIntreg,new Point(-1,-1));
                Image<Gray, float>[] diff = new Image<Gray, float>[9];
                List<float> scoruri = new List<float>();
                for (int image = 0; image < 9; ++image)
                {
                    float scor = 0;
                    diff[image] = new Image<Gray, float>(filterImages[image].Width, filterImages[image].Height);
                    for (int i = 0; i < Math.Min(filtruIntreg.Height, filterImages[image].Height); ++i)
                    {
                        for (int j = 0; j < Math.Min(filtruIntreg.Width, filterImages[image].Width); ++j)
                        {
                            if (filtruIntreg.Data[i, j, 0] == 0 && filterImages[image].Data[i, j, 0] == 0)
                            {
                                diff[image].Data[i, j, 0] = 0;
                                //scor++;
                            }
                            else if (filtruIntreg.Data[i, j, 0] == 0 && filterImages[image].Data[i, j, 0] == 255)
                            {
                                diff[image].Data[i, j, 0] = 0;
                                scor -= 1.0f;
                            }
                            else if (filtruIntreg.Data[i, j, 0] > 0 && filterImages[image].Data[i, j, 0] == 255)
                            {
                                diff[image].Data[i, j, 0] = 255;
                                scor++;
                            }
                            else if (filtruIntreg.Data[i, j, 0] > 0 && filterImages[image].Data[i, j, 0] == 0)
                            {
                                //scor--;
                                diff[image].Data[i, j, 0] = 0;
                            }
                            else
                            {
                                diff[image].Data[i, j, 0] = 150;
                            }
                            //diff[image].Data[i,j,0] = filtruIntreg.Data[i,j,0]*filterImages[image].Data[i,j,0];
                        }
                    }
                    scoruri.Add(scor);
                    //ImageViewer.Show(diff[image]);
                }
                float max = scoruri.Max();
                int found = scoruri.IndexOf(max);
                //ImageViewer.Show(diff[found]);

                maxes.Add(new KeyValuePair<float, Rectangle>(max, subRect[found]));

            }
            float maxOfMaxes = maxes.First().Key;
            int inc = 0;
            int index = 0;
            foreach (KeyValuePair<float, Rectangle> m in maxes)
            {
                if (m.Key > maxOfMaxes)
                {
                    maxOfMaxes = m.Key;
                    index = inc;
                }
                inc++;
            }
            int grosime = 1;
            if (maxOfMaxes > 100)
                grosime = (int)(maxOfMaxes / 100);
            rectangle.Add(new KeyValuePair<int, Rectangle>(grosime, maxes.ElementAt(index).Value));
            return rectangle;
        }
        private static KeyValuePair<int, Rectangle> ApplyFilter(List<Contour<Point>> rectangleContours,
            Image<Gray, Byte> imageCanny, List<Rectangle> candidatesRect, Image<Gray, byte> imageGray)
        {
            KeyValuePair<int, Rectangle> ret = new KeyValuePair<int, Rectangle>();
            List<float> scores = new List<float>();
            foreach (Rectangle candidateRect in candidatesRect)
            {
                int border = (candidateRect.Width + candidateRect.Height) / 7;
                Point startUpperLeft = new Point(candidateRect.X + (border/2),candidateRect.Y + (border/2));
                Point[] centers = new Point[9];
                centers[0] = startUpperLeft;
                centers[1] = new Point(startUpperLeft.X, startUpperLeft.Y + border);
                centers[2] = new Point(startUpperLeft.X, startUpperLeft.Y + (border * 2));

                centers[3] = new Point(startUpperLeft.X + border, startUpperLeft.Y);
                centers[4] = new Point(startUpperLeft.X + border, startUpperLeft.Y + border);
                centers[5] = new Point(startUpperLeft.X + border, startUpperLeft.Y + (border * 2));

                centers[6] = new Point(startUpperLeft.X + (border * 2), startUpperLeft.Y);
                centers[7] = new Point(startUpperLeft.X + (border * 2), startUpperLeft.Y + border);
                centers[8] = new Point(startUpperLeft.X + (border * 2), startUpperLeft.Y + (border * 2));

                double sumOfWidths=0;
                int contoursContained=0;
                int widthDeScazut = 0;
                Image<Gray, byte> filtru = new Image<Gray, byte>(candidateRect.Width, candidateRect.Height, new Gray(0));
                List<Point> sample = new List<Point>();
                foreach (Contour<Point> contour in rectangleContours)
                {
                    if (candidateRect.Contains(contour.GetMinAreaRect().MinAreaRect()))
                    {
                        sumOfWidths += Math.Sqrt(contour.Area);
                        ++contoursContained;
                        
                        filtru.Draw(new LineSegment2D(new Point(contour[0].X - candidateRect.X, contour[0].Y - candidateRect.Y),
                            new Point(contour[1].X - candidateRect.X, contour[1].Y - candidateRect.Y)), new Gray(255), 1);
                        filtru.Draw(new LineSegment2D(new Point(contour[1].X - candidateRect.X, contour[1].Y - candidateRect.Y),
                            new Point(contour[2].X - candidateRect.X, contour[2].Y - candidateRect.Y)), new Gray(255), 1);
                        filtru.Draw(new LineSegment2D(new Point(contour[2].X - candidateRect.X, contour[2].Y - candidateRect.Y),
                            new Point(contour[3].X - candidateRect.X, contour[3].Y - candidateRect.Y)), new Gray(255), 1);
                        filtru.Draw(new LineSegment2D(new Point(contour[3].X - candidateRect.X, contour[3].Y - candidateRect.Y),
                            new Point(contour[0].X - candidateRect.X, contour[0].Y - candidateRect.Y)), new Gray(255), 1);

                        int centerHit = -1;
                        for (int i = 0; i < 9; ++i)
                        {
                            if (contour.InContour(centers[i]) > 0)
                            {
                                centers[i].X = -1000;
                                centers[i].Y = -1000;
                                centerHit = i;
                            }
                        }

                        if (sample.Count == 0 && centerHit>=0)
                        {
                            widthDeScazut = contour.GetMinAreaRect().MinAreaRect().Width;
                            for (int i = 0; i < 4; ++i)
                            {
                                switch (centerHit)
                                {
                                    case 0:
                                        sample.Add(new Point(contour[i].X - candidateRect.X, 
                                            contour[i].Y - candidateRect.Y));
                                        break;
                                    case 1:
                                        sample.Add(new Point(contour[i].X - candidateRect.X,
                                            contour[i].Y - candidateRect.Y-widthDeScazut));
                                        break;
                                    case 2:
                                        sample.Add(new Point(contour[i].X - candidateRect.X,
                                            contour[i].Y - candidateRect.Y - (2*widthDeScazut)));
                                        break;

                                    case 3:
                                        sample.Add(new Point(contour[i].X - candidateRect.X - widthDeScazut, 
                                            contour[i].Y - candidateRect.Y));
                                        break;
                                    case 4:
                                        sample.Add(new Point(contour[i].X - candidateRect.X - widthDeScazut, 
                                            contour[i].Y - candidateRect.Y - widthDeScazut));
                                        break;
                                    case 5:
                                        sample.Add(new Point(contour[i].X - candidateRect.X - widthDeScazut, 
                                            contour[i].Y - candidateRect.Y - (2 * widthDeScazut)));
                                        break;

                                    case 6:
                                        sample.Add(new Point(contour[i].X - candidateRect.X - (2 * widthDeScazut), 
                                            contour[i].Y - candidateRect.Y));
                                        break;
                                    case 7:
                                        sample.Add(new Point(contour[i].X - candidateRect.X - (2 * widthDeScazut), 
                                            contour[i].Y - candidateRect.Y - widthDeScazut));
                                        break;
                                    case 8:
                                        sample.Add(new Point(contour[i].X - candidateRect.X - (2 * widthDeScazut), 
                                            contour[i].Y - candidateRect.Y - (2 * widthDeScazut)));
                                        break;
                                }
                                
                            }
                        }

                    }
                }
                if (contoursContained == 0)
                    continue;
                if (sample.Count == 0)
                    continue;
                int meanWidth = (int)sumOfWidths / contoursContained;
                
                Image<Gray,byte> filtruBlured = new Image<Gray,byte>(filtru.Width,filtru.Height,new Gray(0));
                for (int i = 0; i < 9; ++i)
                {
                    if (centers[i].X == -1000)
                        continue;
                    switch(i)
                    {
                        case 0:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X,sample.ElementAt(0).Y),
                                new Point(sample.ElementAt(1).X,sample.ElementAt(1).Y)),new Gray(255),2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X, sample.ElementAt(1).Y),
                                new Point(sample.ElementAt(2).X, sample.ElementAt(2).Y)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X, sample.ElementAt(2).Y),
                                new Point(sample.ElementAt(3).X, sample.ElementAt(3).Y)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X, sample.ElementAt(3).Y),
                                new Point(sample.ElementAt(0).X, sample.ElementAt(0).Y)), new Gray(255), 2);
                            break;
                        case 1:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X,sample.ElementAt(0).Y + widthDeScazut),
                                new Point(sample.ElementAt(1).X, sample.ElementAt(1).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X, sample.ElementAt(1).Y + widthDeScazut),
                                new Point(sample.ElementAt(2).X, sample.ElementAt(2).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X, sample.ElementAt(2).Y + widthDeScazut),
                                new Point(sample.ElementAt(3).X, sample.ElementAt(3).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X, sample.ElementAt(3).Y + widthDeScazut),
                                new Point(sample.ElementAt(0).X, sample.ElementAt(0).Y + widthDeScazut)), new Gray(255), 2);
                            break;
                        case 2:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X,sample.ElementAt(0).Y+(2*widthDeScazut)),
                                new Point(sample.ElementAt(1).X,sample.ElementAt(1).Y+(2*widthDeScazut))),new Gray(255),2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X, sample.ElementAt(1).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(2).X, sample.ElementAt(2).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X, sample.ElementAt(2).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(3).X, sample.ElementAt(3).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X, sample.ElementAt(3).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(0).X, sample.ElementAt(0).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            break;
                        case 3:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X + widthDeScazut, sample.ElementAt(0).Y),
                                new Point(sample.ElementAt(1).X + widthDeScazut, sample.ElementAt(1).Y)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X + widthDeScazut, sample.ElementAt(1).Y),
                                new Point(sample.ElementAt(2).X + widthDeScazut, sample.ElementAt(2).Y)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X + widthDeScazut, sample.ElementAt(2).Y),
                                new Point(sample.ElementAt(3).X + widthDeScazut, sample.ElementAt(3).Y)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X + widthDeScazut, sample.ElementAt(3).Y),
                                new Point(sample.ElementAt(0).X + widthDeScazut, sample.ElementAt(0).Y)), new Gray(255), 2);
                            break;
                        case 4:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X + widthDeScazut, sample.ElementAt(0).Y + widthDeScazut),
                                new Point(sample.ElementAt(1).X + widthDeScazut, sample.ElementAt(1).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X + widthDeScazut, sample.ElementAt(1).Y + widthDeScazut),
                                new Point(sample.ElementAt(2).X + widthDeScazut, sample.ElementAt(2).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X + widthDeScazut, sample.ElementAt(2).Y + widthDeScazut),
                                new Point(sample.ElementAt(3).X + widthDeScazut, sample.ElementAt(3).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X + widthDeScazut, sample.ElementAt(3).Y + widthDeScazut),
                                new Point(sample.ElementAt(0).X + widthDeScazut, sample.ElementAt(0).Y + widthDeScazut)), new Gray(255), 2);
                            break;
                        case 5:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X + widthDeScazut, sample.ElementAt(0).Y +(2*widthDeScazut)),
                                new Point(sample.ElementAt(1).X + widthDeScazut, sample.ElementAt(1).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X + widthDeScazut, sample.ElementAt(1).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(2).X + widthDeScazut, sample.ElementAt(2).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X + widthDeScazut, sample.ElementAt(2).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(3).X + widthDeScazut, sample.ElementAt(3).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X + widthDeScazut, sample.ElementAt(3).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(0).X + widthDeScazut, sample.ElementAt(0).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            break;
                        case 6:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X + (2 * widthDeScazut), sample.ElementAt(0).Y),
                                new Point(sample.ElementAt(1).X + (2 * widthDeScazut), sample.ElementAt(1).Y)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X + (2 * widthDeScazut), sample.ElementAt(1).Y),
                                new Point(sample.ElementAt(2).X + (2 * widthDeScazut), sample.ElementAt(2).Y)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X + (2 * widthDeScazut), sample.ElementAt(2).Y),
                                new Point(sample.ElementAt(3).X + (2 * widthDeScazut), sample.ElementAt(3).Y)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X + (2 * widthDeScazut), sample.ElementAt(3).Y),
                                new Point(sample.ElementAt(0).X + (2 * widthDeScazut), sample.ElementAt(0).Y)), new Gray(255), 2);
                            break;
                        case 7:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X + (2 * widthDeScazut), sample.ElementAt(0).Y+widthDeScazut),
                                new Point(sample.ElementAt(1).X + (2 * widthDeScazut), sample.ElementAt(1).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X + (2 * widthDeScazut), sample.ElementAt(1).Y + widthDeScazut),
                                new Point(sample.ElementAt(2).X + (2 * widthDeScazut), sample.ElementAt(2).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X + (2 * widthDeScazut), sample.ElementAt(2).Y + widthDeScazut),
                                new Point(sample.ElementAt(3).X + (2 * widthDeScazut), sample.ElementAt(3).Y + widthDeScazut)), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X + (2 * widthDeScazut), sample.ElementAt(3).Y + widthDeScazut),
                                new Point(sample.ElementAt(0).X + (2 * widthDeScazut), sample.ElementAt(0).Y + widthDeScazut)), new Gray(255), 2);
                            break;
                        case 8:
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(0).X + (2 * widthDeScazut), sample.ElementAt(0).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(1).X + (2 * widthDeScazut), sample.ElementAt(1).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(1).X + (2 * widthDeScazut), sample.ElementAt(1).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(2).X + (2 * widthDeScazut), sample.ElementAt(2).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(2).X + (2 * widthDeScazut), sample.ElementAt(2).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(3).X + (2 * widthDeScazut), sample.ElementAt(3).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            filtruBlured.Draw(new LineSegment2D(new Point(sample.ElementAt(3).X + (2 * widthDeScazut), sample.ElementAt(3).Y + (2 * widthDeScazut)),
                                new Point(sample.ElementAt(0).X + (2 * widthDeScazut), sample.ElementAt(0).Y + (2 * widthDeScazut))), new Gray(255), 2);
                            break;
                    }
                   
                }
                //filtruBlured = filtruBlured.SmoothBlur(3,3,true);
                //CvInvoke.cvShowImage("filtru",filtru);
                //CvInvoke.cvShowImage("blur", filtruBlured);
                CvInvoke.cvAdd(filtru, filtruBlured, filtru, new Image<Gray,byte>(filtru.Width,filtru.Height,new Gray(255)));
                //CvInvoke.cvShowImage("sum", filtru);
                CvInvoke.cvSub(new Image<Gray,byte>(filtru.Width,filtru.Height,new Gray(255)),filtru,filtru,new Image<Gray,byte>(filtru.Width,filtru.Height,new Gray(255)));
                Image<Gray, float> distTransf = new Image<Gray, float>(filtru.Size);
                Image<Gray, int> labels = new Image<Gray, int>(filtru.Size);
                CvInvoke.cvDistTransform(filtru.Ptr,distTransf.Ptr,Emgu.CV.CvEnum.DIST_TYPE.CV_DIST_L1,3,null,labels.Ptr);
                Image<Gray, float> distTransfNorm = new Image<Gray, float>(distTransf.Size);
                CvInvoke.cvNormalize(distTransf, distTransfNorm, 0, 1, NORM_TYPE.CV_MINMAX, IntPtr.Zero);
                CvInvoke.cvShowImage("trans",distTransfNorm);
                distTransf.Save("distTransf1.png");
               
                Image<Gray, float> invertedTransf = new Image<Gray, float>(filtru.Size);
                invertedTransf = distTransf * -1;
                for (int i = 0; i < invertedTransf.Height; ++i)
                {
                    for (int j = 0; j < invertedTransf.Width; ++j)
                    {
                        if (invertedTransf.Data[i, j, 0] == 0)
                        {
                            invertedTransf.Data[i, j, 0] = 1;
                        }
                    }
                }

                using (TextWriter tw = new StreamWriter("filtru.txt"))
                {
                    for (int i = 0; i < invertedTransf.Height; i++)
                    {
                        for (int j = 0; j < invertedTransf.Width; j++)
                        {
                            if (j != 0)
                            {
                                tw.Write(" ");
                            }
                            tw.Write(invertedTransf[i, j]);
                        }
                        tw.WriteLine();
                    }
                }

                Image<Gray, byte> cropImage = new Image<Gray, byte>(filtru.Width, filtru.Height);
                
                //!!candidateRect out of bounds imageCanny
                Rectangle candidateRectCopy = candidateRect;
                if (candidateRect.Y+candidateRect.Height > imageCanny.Height)
                    candidateRectCopy = new Rectangle(candidateRect.X, candidateRect.Y, candidateRect.Width, imageCanny.Height-candidateRect.Y);
                if (candidateRect.X+candidateRect.Width > imageCanny.Width)
                    candidateRectCopy = new Rectangle(candidateRect.X, candidateRect.Y, imageCanny.Width-candidateRect.X, candidateRect.Height);
                if (candidateRect.X < 0)
                    candidateRectCopy = new Rectangle(0, candidateRectCopy.Y, candidateRectCopy.Width, candidateRectCopy.Height);
                if (candidateRect.Y < 0)
                    candidateRectCopy = new Rectangle(candidateRectCopy.X, 0, candidateRectCopy.Width, candidateRectCopy.Height);
                cropImage = new Image<Gray, byte>(candidateRectCopy.Width, candidateRectCopy.Height);
                //imageCanny=imageCanny.Dilate(1);
                imageGray.GetSubRect(candidateRectCopy).CopyTo(cropImage);
                Image<Gray, byte> fullMask = new Image<Gray, byte>(cropImage.Width, cropImage.Height, new Gray(180));
                CvInvoke.cvSub(fullMask, cropImage, cropImage, fullMask);
                //CvInvoke.cvShowImage("croped",cropImage);
                float scor = 0;
                Image<Gray, byte> diff = new Image<Gray, byte>(invertedTransf.Width, invertedTransf.Height, new Gray(0));
                //CvInvoke.cvNormalize(cropImage, cropImage, 0, 1, NORM_TYPE.CV_MINMAX, IntPtr.Zero);

                for (int i = 0; i < Math.Min(invertedTransf.Height, cropImage.Height); ++i)
                {
                    for (int j = 0; j < Math.Min(invertedTransf.Width, cropImage.Width); ++j)
                    {
                        /*if (filtru.Data[i, j, 0] == 0 && cropImage.Data[i, j, 0] == 0)
                        {
                            diff.Data[i, j, 0] = 255;
                            scor+=0.3f;
                        }
                        else if (filtru.Data[i, j, 0] == 0 && cropImage.Data[i, j, 0] == 255)
                        {
                            diff.Data[i, j, 0] = 0;
                            scor -= 1f;
                        }
                        else if (filtru.Data[i, j, 0] > 0 && cropImage.Data[i, j, 0] == 255)
                        {
                            diff.Data[i, j, 0] = 255;
                            scor+=1f;
                        }
                        else if (filtru.Data[i, j, 0] > 0 && cropImage.Data[i, j, 0] == 0)
                        {
                            scor-=1.0f;
                            diff.Data[i, j, 0] = 0;
                        }*/
                        //diff[image].Data[i,j,0] = filtruIntreg.Data[i,j,0]*filterImages[image].Data[i,j,0];
                        scor += cropImage.Data[i, j, 0] * invertedTransf.Data[i, j, 0];
                    }
                }
                
                //ImageViewer.Show(diff, "diff");
                
                scores.Add(scor);
                Console.WriteLine(scor);
            }
            //!de verificat inaintea max scores sa fie nenul
            float maxScor = scores.Max();
            //scores.Remove(maxScor);
            //float secondMaxScor = scores.Count > 0 ? scores.Max() : 0 ;
            int r = 0;
            Rectangle secondMaxRect = new Rectangle();
            foreach (float sc in scores)
            {
                if (sc == maxScor)
                {
                    ret = new KeyValuePair<int, Rectangle>(8, candidatesRect.ElementAt(r));
                    
                }
                ++r;
            }
            CvInvoke.cvShowImage("canny",imageCanny);
            return ret;
        }
        private static KeyValuePair<int, Rectangle> SearchForNeighbours(List<Contour<Point>> rectangleContours,
            Image<Gray, Byte> imageCanny, Image<Gray, byte> imageGray)
        {
            KeyValuePair<int, Rectangle> rectangle = new KeyValuePair<int, Rectangle>();
            List<List<int>> scores = new List<List<int>>();
            List<List<Rectangle>> scoresAndRectangles = new List<List<Rectangle>>();

            foreach (Contour<Point> contour in rectangleContours)
            {
                List<Rectangle> addScoresAndRectangles = new List<Rectangle>();

                Point upperLeftRect = new Point(contour.BoundingRectangle.Location.X - (contour.BoundingRectangle.Width / 3), contour.BoundingRectangle.Location.Y - (contour.BoundingRectangle.Height / 3));
                Size rectangleSize = new Size((int)(contour.BoundingRectangle.Width * 3.7), (int)(contour.BoundingRectangle.Height * 3.7));

                Rectangle pos1 = new Rectangle(upperLeftRect, rectangleSize);
                Rectangle pos2 = new Rectangle(upperLeftRect.X - contour.BoundingRectangle.Width, upperLeftRect.Y, rectangleSize.Width, rectangleSize.Height);
                Rectangle pos3 = new Rectangle(upperLeftRect.X - (contour.BoundingRectangle.Width * 2), upperLeftRect.Y, rectangleSize.Width, rectangleSize.Height);

                Rectangle pos4 = new Rectangle(upperLeftRect.X, upperLeftRect.Y - contour.BoundingRectangle.Height, rectangleSize.Width, rectangleSize.Height);
                Rectangle pos5 = new Rectangle(upperLeftRect.X - contour.BoundingRectangle.Width, upperLeftRect.Y - contour.BoundingRectangle.Height, rectangleSize.Width, rectangleSize.Height);
                Rectangle pos6 = new Rectangle(upperLeftRect.X - (contour.BoundingRectangle.Width * 2), upperLeftRect.Y - contour.BoundingRectangle.Height, rectangleSize.Width, rectangleSize.Height);

                Rectangle pos7 = new Rectangle(upperLeftRect.X, upperLeftRect.Y - (contour.BoundingRectangle.Height * 2), rectangleSize.Width, rectangleSize.Height);
                Rectangle pos8 = new Rectangle(upperLeftRect.X - contour.BoundingRectangle.Width, upperLeftRect.Y - (contour.BoundingRectangle.Height * 2), rectangleSize.Width, rectangleSize.Height);
                Rectangle pos9 = new Rectangle(upperLeftRect.X - (contour.BoundingRectangle.Width * 2), upperLeftRect.Y - (contour.BoundingRectangle.Height * 2), rectangleSize.Width, rectangleSize.Height);

                int[] inside = new int[9];

                List<int> currentSquareScores = new List<int>();

                foreach (Contour<Point> other in rectangleContours)
                {
                    if (other.BoundingRectangle.Location == contour.BoundingRectangle.Location)
                        continue;
                    if (pos1.Contains(other.BoundingRectangle))
                    {
                        ++inside[0];
                    }
                    if (pos2.Contains(other.BoundingRectangle))
                    {
                        ++inside[1];
                    }
                    if (pos3.Contains(other.BoundingRectangle))
                    {
                        ++inside[2];
                    }
                    if (pos4.Contains(other.BoundingRectangle))
                    {
                        ++inside[3];
                    }
                    if (pos5.Contains(other.BoundingRectangle))
                    {
                        ++inside[4];
                    }
                    if (pos6.Contains(other.BoundingRectangle))
                    {
                        ++inside[5];
                    }
                    if (pos7.Contains(other.BoundingRectangle))
                    {
                        ++inside[6];
                    }
                    if (pos8.Contains(other.BoundingRectangle))
                    {
                        ++inside[7];
                    }
                    if (pos9.Contains(other.BoundingRectangle))
                    {
                        ++inside[8];
                    }
                }
                int hits = 0;
                for (int i = 0; i < 9; ++i)
                {
                    currentSquareScores.Add(inside[i]);
                    if (hits < inside[i])
                        hits = inside[i];
                }
                Console.WriteLine(hits);
                if (hits == 0)
                {
                    currentSquareScores.Clear();
                    continue;
                }
                scores.Add(currentSquareScores);

                addScoresAndRectangles.Add(pos1);
                addScoresAndRectangles.Add(pos2);
                addScoresAndRectangles.Add(pos3);
                addScoresAndRectangles.Add(pos4);
                addScoresAndRectangles.Add(pos5);
                addScoresAndRectangles.Add(pos6);
                addScoresAndRectangles.Add(pos7);
                addScoresAndRectangles.Add(pos8);
                addScoresAndRectangles.Add(pos9);
                scoresAndRectangles.Add(addScoresAndRectangles);
                if (hits > 5)
                    break;
            }

            int maxScore = 0;
            foreach (List<int> scor in scores)
            {
                if (maxScore < scor.Max())
                {
                    maxScore = scor.Max();
                }
            }

            int j = 0, k = 0;
            List<Rectangle> candidatesRect = new List<Rectangle>();
            foreach (List<int> score in scores)
            {
                k = 0;
                foreach (int val in score)
                {
                    if (val == maxScore)
                    {
                        rectangle = new KeyValuePair<int, Rectangle>(maxScore, scoresAndRectangles.ElementAt(j).ElementAt(k));
                        candidatesRect.Add(scoresAndRectangles.ElementAt(j).ElementAt(k));
                    }
                    ++k;
                }
                ++j;
            }
            if (maxScore > 5)//a detectat cel putin 7 etichete nu e nevoie de filtrare
                return rectangle;
            else if (candidatesRect.Count > 0)
            {
                return ApplyFilter(rectangleContours, imageCanny, candidatesRect, imageGray);
            }
            else
            {
                retry = true;
                return rectangle;
            }
        }
        private static Image<Bgr, Byte> DetectContours(Image<Gray, Byte> imageCanny, Image<Bgr, Byte> imageFrame)
        {
            //ImageViewer.Show(imageCanny);
            List<Contour<Point>> rectangleContours = new List<Contour<Point>>();
            Image<Hsv, byte> imageHsv = imageFrame.Convert<Hsv, Byte>();
            Image<Gray, byte> imageGray = imageHsv[2];
            MemStorage storage = new MemStorage();

            for (Contour<Point> contours = imageCanny.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_NONE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, storage);
                         contours != null; contours = contours.HNext)
            {
                Contour<Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.015, storage);
                if (currentContour.ToArray().Length == 4 && currentContour.Convex
                    && currentContour.Area > 30)
                {
                    rectangleContours.Add(currentContour);
                    CvInvoke.cvDrawContours(imageFrame, currentContour, new MCvScalar(0, 255, 0), new MCvScalar(0, 255, 0), -1, 2, Emgu.CV.CvEnum.LINE_TYPE.EIGHT_CONNECTED, new Point(0, 0));
                }
            }
            //elimina contururile duble si cele careau aria diferita de medie
            List<Contour<Point>> rectangleContoursCopy = new List<Contour<Point>>();
            rectangleContoursCopy.AddRange(rectangleContours);
            double areaSum = 0;
            foreach(Contour<Point> c in rectangleContours)
            {
                areaSum += c.Area;
            }
            double averageArea = areaSum / rectangleContours.Count;
            for (int i = 0; i < rectangleContoursCopy.Count; ++i)
            {
                float raportHeightWidth = (float)rectangleContoursCopy.ElementAt(i).BoundingRectangle.Height / (float)rectangleContoursCopy.ElementAt(i).BoundingRectangle.Width;
                float maxRaportPerimetru=0,minRaportPerimetru=1000;
                double perimetru = rectangleContoursCopy.ElementAt(i).Perimeter/4;
                for(int p=0;p<4;++p)
                {
                    int nextInd = p==3?0:p;
                    float per =(float)Math.Sqrt(((rectangleContoursCopy.ElementAt(i).ElementAt(nextInd).X-rectangleContoursCopy.ElementAt(i).ElementAt(p).X)^2)+
                        ((rectangleContoursCopy.ElementAt(i).ElementAt(nextInd).Y-rectangleContoursCopy.ElementAt(i).ElementAt(p).Y)^2));
                    float rap=per/(float)perimetru;
                    if(maxRaportPerimetru>rap)
                        maxRaportPerimetru=rap;
                    if(minRaportPerimetru<rap)
                        minRaportPerimetru=rap;
                }
                if(raportHeightWidth >1.5 || raportHeightWidth < 0.6)        
                {
                    rectangleContours.Remove(rectangleContoursCopy.ElementAt(i));
                }else if(minRaportPerimetru<0.7 ||maxRaportPerimetru>1.6)
                {
                    rectangleContours.Remove(rectangleContoursCopy.ElementAt(i));
                }
                else if (rectangleContoursCopy.ElementAt(i).Area / averageArea > 2.3 ||
                    rectangleContoursCopy.ElementAt(i).Area / averageArea < 0.2)
                {
                    rectangleContours.Remove(rectangleContoursCopy.ElementAt(i));
                }
            }
            rectangleContoursCopy.Clear();
            rectangleContoursCopy = rectangleContours;
            for (int i = 0; i < rectangleContoursCopy.Count; ++i)
            {
                for (int j = i + 1; j < rectangleContoursCopy.Count; ++j)
                {
                    if (rectangleContoursCopy.ElementAt(i).InContour(new Point(rectangleContoursCopy.ElementAt(j).BoundingRectangle.Location.X + (rectangleContoursCopy.ElementAt(j).BoundingRectangle.Width / 2), rectangleContoursCopy.ElementAt(j).BoundingRectangle.Y + (rectangleContoursCopy.ElementAt(j).BoundingRectangle.Height / 2))) >= 0)
                    {
                        rectangleContours.Remove(rectangleContoursCopy.ElementAt(j));
                    }
                }
            }
            if (rectangleContours.Count <2)
            {
                retry = true;
                return imageFrame;
            }
            KeyValuePair<int, Rectangle> neighbourContours = SearchForNeighbours(rectangleContours, imageCanny, imageGray);

            if (neighbourContours.Key < 6 || retry)
            {
                retry = true;
                return imageFrame;
            }
            
            if (neighbourContours.Value.Width > imageFrame.Width || neighbourContours.Value.Height > imageFrame.Height ||
                neighbourContours.Value.Location.X < 0 || neighbourContours.Value.Location.Y < 0)
            {
                retry = true;
                return imageFrame;
            }
            
            Rectangle detectedRectangleFromImage = new Rectangle(neighbourContours.Value.Location.X,
                neighbourContours.Value.Location.Y,
                neighbourContours.Value.Location.X + neighbourContours.Value.Width > imageFrame.Width ? imageFrame.Width - neighbourContours.Value.Location.X : neighbourContours.Value.Width,
                neighbourContours.Value.Location.Y + neighbourContours.Value.Height > imageFrame.Height ? imageFrame.Height - neighbourContours.Value.Location.Y : neighbourContours.Value.Height);
            Image<Bgr, byte> imageDetected = new Image<Bgr, byte>(detectedRectangleFromImage.Width, detectedRectangleFromImage.Height);

            IdentifyColors(imageFrame, detectedRectangleFromImage, rectangleContours);

            imageFrame.Draw(neighbourContours.Value, new Bgr(0, 0, 255), 2);

            imageFrame.GetSubRect(detectedRectangleFromImage).CopyTo(imageDetected);

            Matrix<float> colors = new Matrix<float>(imageDetected.Rows * imageDetected.Cols, 1, 3);
            Matrix<int> finalClusters = new Matrix<int>(imageDetected.Rows * imageDetected.Cols, 1);

            for (int y = 0; y < imageDetected.Rows; y++)
            {
                for (int x = 0; x < imageDetected.Cols; x++)
                {
                    colors.Data[y + x * imageDetected.Rows, 0] = (float)imageDetected[y, x].Blue;
                    colors.Data[y + x * imageDetected.Rows, 1] = (float)imageDetected[y, x].Green;
                    colors.Data[y + x * imageDetected.Rows, 2] = (float)imageDetected[y, x].Red;
                }
            }
            
            Matrix<Single> centers = new Matrix<Single>(7, colors.Cols, 3);
            CvInvoke.cvKMeans2(colors, 7, finalClusters, new MCvTermCriteria(), 5, IntPtr.Zero, KMeansInitType.PPCenters, centers, IntPtr.Zero);

            Image<Bgr, Byte> imageKmeans = new Image<Bgr, Byte>(imageDetected.Size);

            for (int y = 0; y < imageDetected.Rows; y++)
            {
                for (int x = 0; x < imageDetected.Cols; x++)
                {
                    int cluster_idx = finalClusters[y + x * imageDetected.Rows, 0];
                    MCvScalar sca1 = CvInvoke.cvGet2D(centers, cluster_idx, 0);
                    Bgr color = new Bgr(sca1.v0, sca1.v1, sca1.v2);

                    imageKmeans.Draw(new Rectangle(x,y, 1,1), color, 1);
                }
            }

            //CvInvoke.cvShowImage("kmeans dupa inlocuire", imageKmeans);

            return imageFrame;
        }

        private static Image<Bgr, Byte> DetectSIFT(Image<Gray, Byte> imageGray, Image<Bgr, Byte> imageFrame)
        {
            String modelImageFileName = "D:\\Anul_IV_sem_I\\Rubik\\poze_cub\\face3.jpg";
            Image<Gray, Byte> modelImage = new Image<Gray, byte>(modelImageFileName);
            HomographyMatrix homography = null;
            SURFDetector surfCPU = new SURFDetector(300, false);

            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            Matrix<int> indices;

            Matrix<byte> mask;
            int k = 2;
            double uniquenessThreshold = 0.8;

            modelKeyPoints = surfCPU.DetectKeyPointsRaw(modelImage, null);
            Matrix<float> modelDescriptors = surfCPU.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

            observedKeyPoints = surfCPU.DetectKeyPointsRaw(imageGray, null);
            Matrix<float> observedDescriptors = surfCPU.ComputeDescriptorsRaw(imageGray, null, observedKeyPoints);
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

            Image<Bgr, Byte> modelImage2 = new Image<Bgr, byte>(modelImageFileName);

            Image<Bgr, Byte> result = Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, imageGray, observedKeyPoints,
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
            return result;
        }

        public static void GoCapturing()
        {
            try
            {
                capture = new Capture();
            }
            catch (NullReferenceException ex)
            {
                Console.WriteLine(ex.Message);
            }
            retry = true;
            while (retry) //while nu sunt 4 fete complete
            {
                retry = false;
                Image<Bgr, Byte> imageFrame = capture.QueryFrame();
                Image<Hsv, Byte> imageHsv = imageFrame.Convert<Hsv, Byte>();
                Image<Gray, Byte> imageGray = imageHsv[2];

                //ImageViewer.Show(imageGray);
                Image<Gray, Byte> imageCanny = imageGray.Canny(new Gray(10), new Gray(150));

                Image<Bgr, Byte> imageResult = DetectContours(imageCanny, imageFrame);
                //Image<Bgr, Byte> imageResult = DetectHoughLines(imageCanny, imageFrame);
                //Image<Bgr, Byte> imageResult = DetectSIFT(imageGray, imageFrame);
                if (!retry)
                {
                    imageFrame.Save("Captura2.png");
                    ImageViewer.Show(imageResult, "Capture");
                    imageCanny.Save("canny5.png");
                    imageFrame.Save("frame5.png");
                    System.Threading.Thread.Sleep(100);
                }
            }
        }
        
        public static void IdentifyColors(Image<Bgr, Byte> imageFrame, Rectangle detectedRectangle, List<Contour<Point>> rectangleContour)
        {
            Bgr black = GetBlack(imageFrame, detectedRectangle);
            List<Contour<Point>> insideRectangle = new List<Contour<Point>>();
            Image<Bgr, byte> imageFrameGray = new Image<Bgr, byte>(imageFrame.Width, imageFrame.Height);
            int border = (detectedRectangle.Height + detectedRectangle.Width) / 6;
            int offset = (int)border/2;
            Point startUpperLeft = new Point(detectedRectangle.Location.X+offset,detectedRectangle.Location.Y+offset);
            Point[] centers = new Point[9];
            centers[0] = startUpperLeft;
            centers[1] = new Point(startUpperLeft.X,startUpperLeft.Y+border);
            centers[2] = new Point(startUpperLeft.X,startUpperLeft.Y+(border*2));

            centers[3] = new Point(startUpperLeft.X + border, startUpperLeft.Y);
            centers[4] = new Point(startUpperLeft.X + border, startUpperLeft.Y + border);
            centers[5] = new Point(startUpperLeft.X + border, startUpperLeft.Y + (border * 2));

            centers[6] = new Point(startUpperLeft.X+(border*2),startUpperLeft.Y);
            centers[7] = new Point(startUpperLeft.X+(border*2),startUpperLeft.Y+border);
            centers[8] = new Point(startUpperLeft.X+(border*2),startUpperLeft.Y+(border*2));

            foreach (Contour<Point> c in rectangleContour)
            {
                if (detectedRectangle.Contains(c.BoundingRectangle) &&
                    (c.BoundingRectangle.Height / c.BoundingRectangle.Width) > 0.5 && (c.BoundingRectangle.Height / c.BoundingRectangle.Width) < 1.5)
                {
                    insideRectangle.Add(c);
                    MCvBox2D box = c.GetMinAreaRect();
                    Rectangle minRect = new Rectangle((int)(box.center.X-(box.size.Width/4)),(int)(box.center.Y-(box.size.Height/4)),(int)box.size.Width/2,(int)box.size.Height/2);
                    Image<Bgr, byte> sample = imageFrame.GetSubRect(minRect);
                    Image<Hsv, byte> sampleHsv = sample.Convert<Hsv, byte>();
                    Bgr average = sample.GetAverage();
                    Hsv averageHsv = sampleHsv.GetAverage();

                    average = GetColor(averageHsv,average);
                    
                    imageFrameGray.Draw(minRect, average, 3);
                    for (int i = 0; i < 9; ++i)
                    {
                        if (c.InContour(centers[i])>0)
                        {
                            centers[i].X = -1000;
                            centers[i].Y = -1000;
                        }
                    }
                }
            }
            for (int i = 0; i < 9; ++i)
            {
                if (centers[i].X > 0)
                {
                    Rectangle minRect = new Rectangle(centers[i].X - (int)(border / 3), centers[i].Y - (int)(border / 3), (int)border/2, (int)border/2);
                    Image<Bgr, byte> sample = imageFrame.GetSubRect(minRect);
                    Image<Hsv, byte> sampleHsv = sample.Convert<Hsv, byte>();
                    Bgr average = sample.GetAverage();
                    Hsv averageHsv = sampleHsv.GetAverage();

                    average = GetColor(averageHsv,average);
                    imageFrameGray.Draw(minRect, average, 3);
                }
            }
            CvInvoke.cvShowImage("culori", imageFrameGray);

        }
        private static Bgr GetColor(Hsv hsv, Bgr color)
        {
            Bgr average = new Bgr();
            /*List<double> distances= new List<double>();
            Color[] colors = new Color[6]{Color.Red,Color.LawnGreen,Color.Blue,Color.Yellow,Color.Orange,Color.White};
            distances.Add(255 - color.Red + color.Blue + color.Green);//red
            distances.Add(color.Red + color.Blue + 255 - color.Green);//green
            distances.Add(color.Red + 255 - color.Blue + color.Green);//blue
            distances.Add(200 - color.Red + 200 - color.Green + color.Blue);//yellow
            distances.Add(200 - color.Red + 130 - color.Green + color.Blue);//orange
            distances.Add(200 + 200 + 200 - color.Blue - color.Red - color.Green);//white

            double min = distances.Min(); int i = 0;
            foreach (double val in distances)
            {
                if (val == min)
                {
                    average = new Bgr(colors[i]);
                }
                ++i;
            }*/

            if (hsv.Value > 160 && hsv.Satuation < 40)
            {
                average = new Bgr(Color.White);
            }
            else
            if (hsv.Hue > 110 || hsv.Hue<8)
            {
                average = new Bgr(Color.Red);
            }else
            if (hsv.Hue > 40 && hsv.Hue < 80)
            {
                average = new Bgr(Color.LawnGreen);
            }else
            if (hsv.Hue > 80 && hsv.Hue < 110)
            {
                average = new Bgr(Color.Blue);
            }else
            if (hsv.Hue > 26 && hsv.Hue<40)
            {
                average = new Bgr(Color.Yellow);
            }
            else
            if (hsv.Hue < 26 && hsv.Hue>8)
            {
                average = new Bgr(Color.Orange);
            }
            return average;
        }
        private static Bgr GetBlack(Image<Bgr, byte> imageFrame, Rectangle detectedRectangle)
        {
            imageFrame.Save("1.png");
            int dim = (imageFrame.Height+imageFrame.Width)/50;
            Bgr[] black = new Bgr[dim];
            Bgr trueBlack = new Bgr(0, 0, 0);            
            double [] min,max;
            Point[] minLoc, maxLoc;
            int i = 0;
            int dim2 = dim;
            Image<Bgr, byte> image = imageFrame;
            while(dim2>0)
            {
                image.MinMax(out min, out max, out minLoc, out maxLoc);
                black[i].Blue = image[0].Data[minLoc[0].Y, minLoc[0].X, 0];
                black[i].Green = image[1].Data[minLoc[0].Y, minLoc[0].X, 0];
                black[i].Red = image[2].Data[minLoc[0].Y, minLoc[0].X, 0];
                dim2 -= minLoc.Length;
                for (int j = 0; j < minLoc.Length; ++j)
                {
                    //imageFrame[0].Data[minLoc[j].Y, minLoc[j].X, 0] = 254;
                    //imageFrame[1].Data[minLoc[j].Y, minLoc[j].X, 0] = 254;
                    //imageFrame[2].Data[minLoc[j].Y, minLoc[j].X, 0] = 254;
                    image.Draw(new CircleF(minLoc[j], 1), new Bgr(254, 254, 254), 1);
                }
                ++i;
            }
            double bSum=0,gSum=0,rSum=0;
            for (i = 0; i < dim; ++i)
            {
                bSum += black[i].Blue;
                gSum += black[i].Green;
                rSum += black[i].Red;
            }
            trueBlack.Blue = bSum / dim;
            trueBlack.Green = gSum / dim;
            trueBlack.Red = rSum / dim;

            return trueBlack;
        }
    }
}
