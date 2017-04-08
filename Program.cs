//----------------------------------------------------------------------------
//  Copyright (C) 2004-2012 by EMGU. All rights reserved.       
//----------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.GPU;

namespace SURFFeature2
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            long matchTime;

            String modelImageFileName="D:\\Anul_IV_sem_I\\Rubik\\poze_cub\\7.jpg";
            String observedImageFileName="D:\\Anul_IV_sem_I\\Rubik\\poze_cub\\7.jpg";

            //Image<Bgr, byte> result = DrawMatches.Draw(modelImageFileName,observedImageFileName, out matchTime);
            //Image<Bgr, byte> result = DrawMatches.DrawLines(modelImageFileName, observedImageFileName, out matchTime);
            //Image<Bgr, byte> result = DrawMatches.Parallelogram(modelImageFileName, observedImageFileName, out matchTime);
            
           // ImageViewer.Show(result, String.Format("Matched in {0} milliseconds", matchTime));
           // ImageViewer.Show(new Image<Bgr,byte>(observedImageFileName),"Imagine originala"); 

            CameraCapture.GoCapturing();
        }
    }
}
