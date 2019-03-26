# Onderzoek

## Inhoudstafel

<!-- vim-markdown-toc GitLab -->

* [Detector](#detector)
    * [Keuze van soort model](#keuze-van-soort-model)
    * [Type van architectuur model](#type-van-architectuur-model)
    * [Soorten detectoren](#soorten-detectoren)
    * [Termen](#termen)
* [Sliding Window](#sliding-window)
* [Uitwerking](#uitwerking)
    * [Mappenstructuur](#mappenstructuur)
    * [Training](#training)
* [Problemen](#problemen)
    * [YOLOv3](#yolov3)
    * [Nacht](#nacht)

<!-- vim-markdown-toc -->

## Detector

Detector bestaat uit een classificatiemodel en een localisator model. De eigenlijke uitkomst van het neuraal netwerk is echter niet alleen de klasse van het gedetecteerd object maar alsook de locatie van het object. Dit is een complexe architectuur.

Bij het maken van een object detector zijn er natuurlijk enkele afwegingen te maken zoals welk type van model en welk soort model.

### Keuze van soort model

| Zelf een model trainen       | Een voorgetraind model hertrainen                       | Gebruik maken van voorgetrainde modellen |
| ---------------------------- | ------------------------------------------------------- | ---------------------------------------- |
| Flexibeler                   | Flexibeler                                              | Sneller om op te zetten                  |
| Enorm veel tijd innemen      | Veel tijd innemen                                       | Geen trainingstijd nodig                 |
| Meer data nodig              | Minder data nodig                                       | Geen data nodig                          |
| Zelf de patronen laten leren | Patronen zijn al grotendeels herkend                    | Patronen zijn al grotendeels herkend     |
| Zelf klasses kunnen kiezen   | Zelf klasses kunnen kiezen bovenop de voorgedefinieerde | Klasses zijn voorgedefinieerd            |

**Conclusie:** zelf een model van scratch maken wordt niet gedaan. Deze vergt teveel tijd en de alternatieven zijn gewoonweg beter. Een voorgetraind netwerk gebruiken is een goede keuze wanneer je objecten moet detecteren die niet in de standaard klasses zitten. Je moet hier wel rekening houden dat het label werk en het trainingswerk wel arbeidsintensief zijn. Ik zal echter voor een voorgetraind netwerk kiezen. Mijn focus ligt op het maken van een persoonsdetector, de klasse persoon komt in de meest courante modellen voor en werken zeer goed.

### Type van architectuur model

|  Algorithm  | Features                                                                                                                                                                                       | Prediction time / image | Limitations                                                                                                                                                             |
| :---------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    _CNN_    | Divides the image into multiple regions and then classify each region into various classes.                                                                                                    |            –            | Needs a lot of regions to predict accurately and hence high computation time.                                                                                           |
|     RCNN    | Uses selective search to generate regions. Extracts around 2000 regions from each image.                                                                                                       |      40-50 seconds      | High computation time as each region is passed to the CNN separately also it uses three different model for making predictions.                                         |
|  Fast RCNN  | Each image is passed only once to the CNN and feature maps are extracted. Selective search is used on these maps to generate predictions. Combines all the three models used in RCNN together. |        2 seconds        | Selective search is slow and hence computation time is still high.                                                                                                      |
| Faster RCNN | Replaces the selective search method with region proposal network which made the algorithm much faster.                                                                                        |       0.2 seconds       | Object proposal takes time and as there are different systems working one after the other, the performance of systems depends on how the previous system has performed. |

Bron: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2018/10/a-step-by-step-introduction-to-the-basic-object-detection-algorithms-part-1/)

**Conclusie:** MobileNet is gemaakt voor mobiele en niet krachtige toestellen, het voordeel hierbij dat deze snel en compact zijn, qua accuraatheid is MobileNet niet optimaal. Inception is dan weer beter voor iets krachtigere toestellen zoals een laptop, de precisie van Inception is stukken beter. Als laatste is ResNet te complex waardoor het trager werkt. Uit mijn eigen onderzoek kwam Inception er het best uit.

### Soorten detectoren

-   SSD
-   R-CNN
-   Faster R-CNN
-   Mask R-CNN
-   Yolo

### Termen

-   Non Maxiumum Supression (NMS)

Meerdere detections samenvoegen. Neemt de detectie met de grootste probabiliteit en supress de andere detections die overlappen. Je geeft enkel de grootste probabiliteit van detectie weer. Dit wordt gebruikt bij het detecteren ieder object apart.

-   Intersection of Union (IoU)

![Computing the Intersection of Union is as simple as dividing the area of overlap between the bounding boxes by the area of union](https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png "graph IoU")

Dit wordt berekend bij een detectie van een object tussen de predictie en de hand gelabelde detectie. Rate van overlap, hoe hoger hoe meer overlap, hoe beter de predictie.

## Sliding Window

Afweging maken tussen hoeveelheid regions (ROI) en computational complexity.

## Uitwerking

### Mappenstructuur

-   Root
    -   images
        -   img1.jpg
        -   img2.jpg
    -   annotations
        -   xmls
            -   img1.xml
            -   img2.xml
        -   label_map.pbtxt
        -   trainval.txt


-   xmls: data met locatie van uw targets, naamgeving zelfde als images
-   label_map.pbtx: verschillende targets een id geven, json
-   trainval: naam van de verschillende images

### Training

1.  Extraheren van personen uit de archief video
2.  MPV gebruiken met trimming script
    -   Verschillende stukken video converten naar images
    -   Iedere image annoteren
    -   Converteren naar TFRecords
3.  Annotatie
    -   LabelImg
    -   XML
4.  Converteren annotaties naar TFRecords
5.  Augmentatie
6.  Model opbouwen

## Problemen

### YOLOv3

-   State of the art object detectie algoritme bovenop Darkflow die geschreven is in C. Probleem hierbij is dat Tensorflow API werkt het best via Python. Converteren van het model is niet ondersteund.

### Nacht

-   Infrarood zorgt voor grijsbeelden.
-   Confident level zo laag houden om iets van detecties te hebben
