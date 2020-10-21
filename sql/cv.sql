select distinct * from Posts where Id<=63760928 and
(Tags like '%<computer-vision>%' or Tags like '%<image-segmentation>%' or Tags like '%<camera-calibration>%' or Tags like '%<matlab-cvst>%'
or Tags like '%<object-detection>%' or Tags like '%<feature-detection>%' or Tags like '%<yolo>%' or Tags like '%<sift>%'
or Tags like '%<image-recognition>%' or Tags like '%<opencv>%' or Tags like '%<image-processing>%' or Tags like '%<emgucv>%'
or Tags like '%<ocr>%' or Tags like '%<mat>%' or Tags like '%<opencv3.0>%' or Tags like '%<javacv>%')
and CreationDate>'2012-01-01';