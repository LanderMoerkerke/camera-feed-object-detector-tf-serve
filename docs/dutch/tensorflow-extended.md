# TensorFlow Extended

## Inhoudstafel

<!-- vim-markdown-toc GitLab -->

* [Intro](#intro)
* [TensorFlow Data Validation](#tensorflow-data-validation)
    * [Voordelen](#voordelen)
    * [Nadelen](#nadelen)
    * [Nuttig voor object detectie](#nuttig-voor-object-detectie)
* [TensorFlow Transform](#tensorflow-transform)
    * [Voordelen](#voordelen-1)
    * [Nadelen](#nadelen-1)
    * [Nuttig voor object detectie](#nuttig-voor-object-detectie-1)
* [TensorFlow Model Analysis](#tensorflow-model-analysis)
    * [Voordelen](#voordelen-2)
    * [Nadelen](#nadelen-2)
    * [Nuttig voor object detectie](#nuttig-voor-object-detectie-2)
* [TensorFlow Serve](#tensorflow-serve)
    * [Voordelen](#voordelen-3)
    * [Nadelen](#nadelen-3)
    * [Nuttig voor object detectie](#nuttig-voor-object-detectie-3)
    * [Keuze gRPC of REST](#keuze-grpc-of-rest)
* [Termen](#termen)
    * [Data drift](#data-drift)
    * [Training-serving skew](#training-serving-skew)
    * [Apache Beam](#apache-beam)
* [Issues](#issues)

<!-- vim-markdown-toc -->

## Intro

TensorFlow Extended is een machine learning platform, die al jaren wordt gebruikt door Google. Sinds 2017 worden er stelselmatig delen van deze workflow opengesteld naar het publiek. Op dit excact moment zijn er vier delen beschikbaar:

-   TensorFlow Data Validation
-   TensorFlow Transform
-   TensorFlow Model Analysis
-   TensorFlow Serving

Het doel van TensorFlow Extended is het minimalizeren van eigen geschreven code en het sneller op punt stellen van een project. Dit zorgt voor snellere en stabielere release cycli.

Bron: [TensorFlow Extended paper](https://dl.acm.org/citation.cfm?id=3098021)

## TensorFlow Data Validation

TensorFlow Data Validation is een open-source library dat zorgt voor ontdekken, analyseren en valideren van data.

Wanneer je data moet analyseren kan je dit nog op kleine schaal zelf implementeren maar naarmate de hoeveelheid data stijgt is dit quasi onmogelijk en een zeer tijdrovend proces. Bij grote hoeveelheden data wordt analyse geschaald over meerdere nodes, TensorFlow Data Validation speelt hier perfect op in.

TFDV gebruikt Apache Beam voor zijn datapipelines, wat voor snelle batch en stream processing zorgt. Het grote nadeel van van het gebruik van Apache Beam is dat de Python API momenteel enkel versie 2 ondersteund.

Wat is TFDV:

-   Detectie anomalieën
-   Visualiseren van de dataset
-   Detecteren van data drift en skew
-   Kijken of training, testing en serving datasets niet te veel van elkaar afwijken

### Voordelen

-   Connecties met Apache Beam
-   Flexibel
-   Schaalbaar
-   Exporteerbaar, kan gebruikt worden door TensorFlow Transform
-   Combineerbaar met nieuwe data
-   Controle op dataset

### Nadelen

-   Python 2
-   Werkt enkel op "big data"

### Nuttig voor object detectie

Niet echt, doordat TFDV vooral focust op het berekenen van beschrijvende statistiek is het toepassing bij object detectie geen meerwaarde. Bij een detector werken we met afbeeldingen hierbij is geven gemiddeldes en standaardafwijkingen geen bruikbare informatie.

## TensorFlow Transform

TensorFlow Transform is een open-source library voor een preprocessing pipeline.

Via een pipeline normalizeerd, converteert het platform data die daarna verder verwerkt kan worden. Het uiteindelijke doel is om deze in te laden in een machine learning model, maar is niet vereist.

Doordat je deze acties kunt exporteren kan je deze gebruiken tijdens training en serving, zo vermijd je training-serving skew. In de graph zitten al de berekeningen van je preprocessing, deze graph gebruik je zowel tijdens de training en de serving.

### Voordelen

-   Kans op training serving skew verminderen
-   Exporteren als graph (!)

### Nadelen

-   Python 2, door dat Apache Beam Python 3 niet ondersteund

### Nuttig voor object detectie

Op zich geen echte meerwaarde, we werken bij een object detectiesysteem niet met numerieke waarden maar met afbeeldingen. Tijdens het trainen en het serven van een model wordt er maar één berekening op de afbeeldingen uitgevoerd, normaliseren. De toepassing van TensorFlow Transform ligt vooral op het gebruik van meerdere calculaties.

## TensorFlow Model Analysis

![Demo](https://www.tensorflow.org/tfx/model_analysis/images/tfma-slicing-metrics-browser.gif)

TensorFlow Model Analysis is een tool voor het evalueren van getrainde TensorFlow modellen. Door middel van Apache Beam wordt het model gedistribueerd getest op veel data. TensorFlow Model Analysis zal dan metrics berekenen die de gebruiker kan raadplegen.

### Voordelen

-   Gedistribueerd
-   Grafisch weergegeven

### Nadelen

-   Werkt enkel op TensorFlow modellen
-   Python 2, door dat Apache Beam Python 3 niet ondersteund

### Nuttig voor object detectie

Ja, TensorFlow Model Analysis kan gebruikt worden op ieder TensorFlow model. De kracht van TFMA ligt hem vooral bij het gebruik van veel data om het model te testen. Om dit goed te doen maak je een aparte dataset om uw model te evalueren, deze moet verschillen van de trainingdata. Je zou het script kunnen uitbreiden zodat deze ook de snelheid van het model controleert. Dit vergt veel tijd maar is wel een meerwaarde bij grotere projecten.

## TensorFlow Serve

TensorFlow Serving is een open-source library voor het in productie stellen van modellen aan de hand van een server.

Via REST of gRPC calls kan een client data sturen naar TensorFlow Serving, deze zal de data dan via de gekozen actie verwerken en terugsturen. Via TensorFlow Serving is het mogelijk om meerdere versies van eenzelfde model in te laden. Zo kan je gemakkelijk een nieuwe versie toevoegen of terug verwijderen.

Het opstellen van deze server gebeurt best via een Docker container, dit geeft de flexibiliteit om deze op ieder medium te laten draaien. Indien de requests te hoog zijn en je Docker conainer sputtert, kan je deze ook via Kubernetes laten schalen.

Het grote voordeel van TensorFlow Serving is dat het model en de code die het model aanroept op verschillende servers staan. Dit zorgt ervoor dat wanneer je veel requests krijgt, je de model server kan laten schalen zonder dat de uitvoerende code - een website of dergelijke - ook mee
schaalt.

### Voordelen

-   Gedecentraliseerd, model staat los van logica
-   Versie beheer
-   REST API, zorgt ervoor dat dit op iedere service werkt
-   Schaalbaar
-   Minder eigen code

### Nadelen

-   Omslachtig exporteren / converteren van een model
-   Duurt even voor op te stellen
-   Latency

### Nuttig voor object detectie

Zeker en vast, TensorFlow Serving kan op ieder model toegepast worden. TF Serve zorgt ervoor dat voor productie omgevingen, TensorFlow de keuze is voor een schaalbaar en betrouwbaar systeem. We sturen een REST request naar de TF Serving model, in de body zit de afbeelding. De request wordt op de andere server verwerkt, daarna wordt het resultaat, de predictie teruggestuurd.

### Keuze gRPC of REST

Bij het gebruik van simpele, niet complexe data, zoals zwart-wit afbeelding of een string is het best op gRPC te gebruiken. gRPC gebruikt minder bandbreedte. REST is dan weer beter bij meer complexe data, de request zelf is wel groter.

Bron: [TensorFlow Serving REST vs gRPC - Medium](https://medium.com/@avidaneran/tensorflow-serving-rest-vs-grpc-e8cef9d4ff62).

## Termen

### Data drift

Data drift is a natural consequence of the diversity of big data sources. The operation, maintenance and modernization of these systems causes unpredictable, unannounced and unending mutations of data characteristics.

Bron: [CMS Wire](https://www.cmswire.com/big-data/big-datas-hidden-scourge-data-drift/)

### Training-serving skew

Doordat training gebeurt op historische batch processing en predictie op stream processing werk je met twee verschillende systeem om data aan te leveren aan het model. Hierbij kan het gebeuren dat de verwerkte data anders is bij beide systemen. Dit noemt men training en serving skew.

### Apache Beam

## Issues

-   [Convert naar TF Serving](https://github.com/tensorflow/models/issues/1988)
-   [Data loss issue](https://github.com/tensorflow/models/issues/2675)
-   [No versions](https://stackoverflow.com/questions/45544928/tensorflow-serving-no-versions-of-servable-model-found-under-base-path/46047081#46047081)
-   [Inference Graph](https://github.com/tensorflow/models/issues/2861)
