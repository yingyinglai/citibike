

```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import evaluation
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
```


```python
from pyspark.sql import SparkSession

# @hidden_cell
# This function is used to setup the access of Spark to your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def set_hadoop_config_with_credentials_6f87b1d25f3b4023bbd4eb9a54f4fa7c(name):
    """This function sets the Hadoop configuration so it is possible to
    access data from Bluemix Object Storage using Spark"""

    prefix = 'fs.swift.service.' + name
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + '.auth.url', 'https://identity.open.softlayer.com'+'/v3/auth/tokens')
    hconf.set(prefix + '.auth.endpoint.prefix', 'endpoints')
    hconf.set(prefix + '.tenant', '5bf68650a13f47d9a10fb4d0f20e1c05')
    hconf.set(prefix + '.username', 'f2e695ab5f6e43d3a7e569148c5cd27a')
    hconf.set(prefix + '.password', 'Xi(3#zNmtL51Td[T')
    hconf.setInt(prefix + '.http.port', 8080)
    hconf.set(prefix + '.region', 'dallas')
    hconf.setBoolean(prefix + '.public', False)

# you can choose any name
name = 'keystone'
set_hadoop_config_with_credentials_6f87b1d25f3b4023bbd4eb9a54f4fa7c(name)

spark = SparkSession.builder.getOrCreate()

df_data_1 = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load('swift://bikeshare.' + name + '/all_keywords.csv')
df_data_1.take(5)
df_data_2 = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load('swift://bikeshare.' + name + '/all_keywords.csv')
df_data_2.take(5)

```




    [Row(Trip Duration=u'680', Start Time=u'2017-01-01 00:00:21', Stop Time=u'2017-01-01 00:11:41', Start Station ID=u'3226', Start Station Name=u'W 82 St & Central Park West', Start Station Latitude=u'40.78275', Start Station Longitude=u'-73.97137', End Station ID=u'3165', End Station Name=u'Central Park West & W 72 St', End Station Latitude=u'40.77579376683666', End Station Longitude=u'-73.9762057363987', Bike ID=u'25542', User Type=u'Subscriber', Birth Year=u'1965', Gender=u'2'),
     Row(Trip Duration=u'1282', Start Time=u'2017-01-01 00:00:45', Stop Time=u'2017-01-01 00:22:08', Start Station ID=u'3263', Start Station Name=u'Cooper Square & E 7 St', Start Station Latitude=u'40.72923649910006', Start Station Longitude=u'-73.99086803197861', End Station ID=u'498', End Station Name=u'Broadway & W 32 St', End Station Latitude=u'40.74854862', End Station Longitude=u'-73.98808416', Bike ID=u'21136', User Type=u'Subscriber', Birth Year=u'1987', Gender=u'2'),
     Row(Trip Duration=u'648', Start Time=u'2017-01-01 00:00:57', Stop Time=u'2017-01-01 00:11:46', Start Station ID=u'3143', Start Station Name=u'5 Ave & E 78 St', Start Station Latitude=u'40.77682863439968', Start Station Longitude=u'-73.96388769149779', End Station ID=u'3152', End Station Name=u'3 Ave & E 71 St', End Station Latitude=u'40.76873687', End Station Longitude=u'-73.96119945', Bike ID=u'18147', User Type=u'Customer', Birth Year=None, Gender=u'0'),
     Row(Trip Duration=u'631', Start Time=u'2017-01-01 00:01:10', Stop Time=u'2017-01-01 00:11:42', Start Station ID=u'3143', Start Station Name=u'5 Ave & E 78 St', Start Station Latitude=u'40.77682863439968', Start Station Longitude=u'-73.96388769149779', End Station ID=u'3152', End Station Name=u'3 Ave & E 71 St', End Station Latitude=u'40.76873687', End Station Longitude=u'-73.96119945', Bike ID=u'21211', User Type=u'Customer', Birth Year=None, Gender=u'0'),
     Row(Trip Duration=u'621', Start Time=u'2017-01-01 00:01:25', Stop Time=u'2017-01-01 00:11:47', Start Station ID=u'3143', Start Station Name=u'5 Ave & E 78 St', Start Station Latitude=u'40.77682863439968', Start Station Longitude=u'-73.96388769149779', End Station ID=u'3152', End Station Name=u'3 Ave & E 71 St', End Station Latitude=u'40.76873687', End Station Longitude=u'-73.96119945', Bike ID=u'26819', User Type=u'Customer', Birth Year=None, Gender=u'0')]




```python
bikeshare = spark.read.csv('swift://bikeshare.' + name + '/all_keywords.csv', header="true", inferSchema="true",mode="DROPMALFORMED")
```


```python
bikeshare.show(5)
```

    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |Trip Duration|         Start Time|          Stop Time|Start Station ID|  Start Station Name|Start Station Latitude|Start Station Longitude|End Station ID|    End Station Name|End Station Latitude|End Station Longitude|Bike ID| User Type|Birth Year|Gender|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |          680|2017-01-01 00:00:21|2017-01-01 00:11:41|            3226|W 82 St & Central...|              40.78275|              -73.97137|          3165|Central Park West...|   40.77579376683666|    -73.9762057363987|  25542|Subscriber|      1965|     2|
    |         1282|2017-01-01 00:00:45|2017-01-01 00:22:08|            3263|Cooper Square & E...|     40.72923649910006|     -73.99086803197861|           498|  Broadway & W 32 St|         40.74854862|         -73.98808416|  21136|Subscriber|      1987|     2|
    |          648|2017-01-01 00:00:57|2017-01-01 00:11:46|            3143|     5 Ave & E 78 St|     40.77682863439968|     -73.96388769149779|          3152|     3 Ave & E 71 St|         40.76873687|         -73.96119945|  18147|  Customer|      null|     0|
    |          631|2017-01-01 00:01:10|2017-01-01 00:11:42|            3143|     5 Ave & E 78 St|     40.77682863439968|     -73.96388769149779|          3152|     3 Ave & E 71 St|         40.76873687|         -73.96119945|  21211|  Customer|      null|     0|
    |          621|2017-01-01 00:01:25|2017-01-01 00:11:47|            3143|     5 Ave & E 78 St|     40.77682863439968|     -73.96388769149779|          3152|     3 Ave & E 71 St|         40.76873687|         -73.96119945|  26819|  Customer|      null|     0|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    only showing top 5 rows
    



```python
assembler = VectorAssembler(inputCols = ["Start Station Latitude","Start Station Longitude"], outputCol="features")
train = assembler.transform(bikeshare)

knum = 5
kmeans = KMeans(featuresCol=assembler.getOutputCol(), predictionCol="cluster", k=knum, seed=456789)
model = kmeans.fit(train)
print "Model Created!"
```

    Model Created!



```python
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
```

    Cluster Centers: 
    [ 40.77245704 -73.96883263]
    [ 0.  0.]
    [ 40.72173041 -73.99739337]
    [ 40.69445825 -73.97267071]
    [ 40.74673798 -73.98867716]



```python
prediction = model.transform(train)
prediction.groupBy("cluster").count().orderBy("cluster").show()
```

    +-------+------+
    |cluster| count|
    +-------+------+
    |      0|414333|
    |      1|     2|
    |      2|633549|
    |      3|319574|
    |      4|878530|
    +-------+------+
    



```python
prediction.select("Start Station ID", "cluster").show(50)
```

    +----------------+-------+
    |Start Station ID|cluster|
    +----------------+-------+
    |            3226|      0|
    |            3263|      2|
    |            3143|      0|
    |            3143|      0|
    |            3143|      0|
    |            3163|      0|
    |             499|      0|
    |             362|      4|
    |             430|      3|
    |            3165|      0|
    |             528|      4|
    |             474|      4|
    |             524|      4|
    |             297|      4|
    |             515|      4|
    |            3172|      0|
    |             515|      4|
    |             423|      4|
    |             267|      4|
    |             369|      2|
    |            3139|      0|
    |             279|      2|
    |             301|      2|
    |             128|      2|
    |             449|      4|
    |             449|      4|
    |             449|      4|
    |             153|      4|
    |             368|      2|
    |             513|      0|
    |             449|      4|
    |             449|      4|
    |             523|      4|
    |            3255|      4|
    |             523|      4|
    |             482|      4|
    |            3429|      3|
    |             473|      2|
    |             516|      0|
    |             472|      4|
    |             472|      4|
    |            3349|      3|
    |            3349|      3|
    |            3349|      3|
    |             478|      4|
    |             442|      4|
    |             466|      4|
    |             442|      4|
    |            3349|      3|
    |            3349|      3|
    +----------------+-------+
    only showing top 50 rows
    



```python
customerCluster = {}
for i in range(0,knum):
    tmp = prediction.select("Trip Duration", "Start Time", "Stop Time", "Start Station ID", "Start Station Name", \
                                        "Start Station Latitude", "Start Station Longitude", "End Station ID", "End Station Name", "End Station Latitude","End Station Longitude","Bike ID","User Type","Birth Year","Gender")\
                                    .where("cluster =" +  str(i))
    customerCluster[str(i)]= tmp
    print "Cluster"+str(i)
    customerCluster[str(i)].show(10)
```

    Cluster0
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |Trip Duration|         Start Time|          Stop Time|Start Station ID|  Start Station Name|Start Station Latitude|Start Station Longitude|End Station ID|    End Station Name|End Station Latitude|End Station Longitude|Bike ID| User Type|Birth Year|Gender|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |          680|2017-01-01 00:00:21|2017-01-01 00:11:41|            3226|W 82 St & Central...|              40.78275|              -73.97137|          3165|Central Park West...|   40.77579376683666|    -73.9762057363987|  25542|Subscriber|      1965|     2|
    |          648|2017-01-01 00:00:57|2017-01-01 00:11:46|            3143|     5 Ave & E 78 St|     40.77682863439968|     -73.96388769149779|          3152|     3 Ave & E 71 St|         40.76873687|         -73.96119945|  18147|  Customer|      null|     0|
    |          631|2017-01-01 00:01:10|2017-01-01 00:11:42|            3143|     5 Ave & E 78 St|     40.77682863439968|     -73.96388769149779|          3152|     3 Ave & E 71 St|         40.76873687|         -73.96119945|  21211|  Customer|      null|     0|
    |          621|2017-01-01 00:01:25|2017-01-01 00:11:47|            3143|     5 Ave & E 78 St|     40.77682863439968|     -73.96388769149779|          3152|     3 Ave & E 71 St|         40.76873687|         -73.96119945|  26819|  Customer|      null|     0|
    |          666|2017-01-01 00:01:51|2017-01-01 00:12:57|            3163|Central Park West...|            40.7734066|           -73.97782542|          3163|Central Park West...|          40.7734066|         -73.97782542|  16050|Subscriber|      2000|     1|
    |          559|2017-01-01 00:05:00|2017-01-01 00:14:20|             499|  Broadway & W 60 St|           40.76915505|           -73.98191841|           479|     9 Ave & W 45 St|         40.76019252|          -73.9912551|  27294|Subscriber|      1973|     1|
    |          634|2017-01-01 00:07:34|2017-01-01 00:18:08|            3165|Central Park West...|     40.77579376683666|      -73.9762057363987|          3164|Columbus Ave & W ...|          40.7770575|         -73.97898475|  16311|Subscriber|      1980|     1|
    |         1246|2017-01-01 00:09:19|2017-01-01 00:30:06|            3172|W 74 St & Columbu...|            40.7785669|           -73.97754961|          3163|Central Park West...|          40.7734066|         -73.97782542|  20845|Subscriber|      1977|     1|
    |          351|2017-01-01 00:10:11|2017-01-01 00:16:02|            3139|  E 72 St & Park Ave|     40.77118287540658|     -73.96409422159195|          3146|     E 81 St & 3 Ave|         40.77573034|          -73.9567526|  26491|Subscriber|      1984|     2|
    |         1386|2017-01-01 00:12:55|2017-01-01 00:36:02|             513|    W 56 St & 10 Ave|             40.768254|             -73.988639|           494|     W 26 St & 8 Ave|         40.74734825|         -73.99723551|  18569|Subscriber|      1986|     1|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    only showing top 10 rows
    
    Cluster1
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |Trip Duration|         Start Time|          Stop Time|Start Station ID|  Start Station Name|Start Station Latitude|Start Station Longitude|End Station ID|    End Station Name|End Station Latitude|End Station Longitude|Bike ID| User Type|Birth Year|Gender|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |          100|2017-01-30 17:36:54|2017-01-30 17:38:34|            3446|NYCBS Depot - STY...|                   0.0|                    0.0|          3446|NYCBS Depot - STY...|                 0.0|                  0.0|  24956|Subscriber|      1987|     1|
    |          271|2017-01-30 19:14:48|2017-01-30 19:19:19|            3446|NYCBS Depot - STY...|                   0.0|                    0.0|          3446|NYCBS Depot - STY...|                 0.0|                  0.0|  25505|Subscriber|      1992|     1|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    
    Cluster2
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |Trip Duration|         Start Time|          Stop Time|Start Station ID|  Start Station Name|Start Station Latitude|Start Station Longitude|End Station ID|    End Station Name|End Station Latitude|End Station Longitude|Bike ID| User Type|Birth Year|Gender|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |         1282|2017-01-01 00:00:45|2017-01-01 00:22:08|            3263|Cooper Square & E...|     40.72923649910006|     -73.99086803197861|           498|  Broadway & W 32 St|         40.74854862|         -73.98808416|  21136|Subscriber|      1987|     2|
    |         1231|2017-01-01 00:09:59|2017-01-01 00:30:31|             369|Washington Pl & 6...|           40.73224119|           -74.00026394|           306|Cliff St & Fulton St|         40.70823502|         -74.00530063|  21128|Subscriber|      1969|     1|
    |          229|2017-01-01 00:10:50|2017-01-01 00:14:40|             279|Peck Slip & Front St|             40.707873|              -74.00167|           308|St James Pl & Oli...|         40.71307916|         -73.99851193|  18417|Subscriber|      1974|     1|
    |          570|2017-01-01 00:10:50|2017-01-01 00:20:20|             301|   E 2 St & Avenue B|           40.72217444|           -73.98368779|           301|   E 2 St & Avenue B|         40.72217444|         -73.98368779|  21400|Subscriber|      1996|     1|
    |          207|2017-01-01 00:10:53|2017-01-01 00:14:20|             128|MacDougal St & Pr...|           40.72710258|           -74.00297088|           229|      Great Jones St|         40.72743423|         -73.99379025|  26577|Subscriber|      1991|     1|
    |         1173|2017-01-01 00:12:40|2017-01-01 00:32:14|             368|  Carmine St & 6 Ave|           40.73038599|           -74.00214988|           524|     W 43 St & 6 Ave|         40.75527307|         -73.98316936|  26003|Subscriber|      1991|     1|
    |          337|2017-01-01 00:14:32|2017-01-01 00:20:09|             473|Rivington St & Ch...|           40.72110063|            -73.9919254|           410|Suffolk St & Stan...|         40.72066442|         -73.98517977|  17108|Subscriber|      1989|     1|
    |         2502|2017-01-01 00:18:12|2017-01-01 00:59:55|            3263|Cooper Square & E...|     40.72923649910006|     -73.99086803197861|           412|Forsyth St & Cana...|          40.7158155|         -73.99422366|  16981|Subscriber|      1988|     1|
    |         1730|2017-01-01 00:19:12|2017-01-01 00:48:03|             312|Allen St & Stanto...|             40.722055|             -73.989111|            82|St James Pl & Pea...|         40.71117416|         -74.00016545|  22906|Subscriber|      1997|     1|
    |         1169|2017-01-01 00:19:39|2017-01-01 00:39:09|             445|  E 10 St & Avenue A|           40.72740794|           -73.98142006|          3243|     E 58 St & 1 Ave|   40.75892386377695|   -73.96226227283478|  23288|Subscriber|      1992|     1|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    only showing top 10 rows
    
    Cluster3
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |Trip Duration|         Start Time|          Stop Time|Start Station ID|  Start Station Name|Start Station Latitude|Start Station Longitude|End Station ID|    End Station Name|End Station Latitude|End Station Longitude|Bike ID| User Type|Birth Year|Gender|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |          255|2017-01-01 00:05:47|2017-01-01 00:10:02|             430|    York St & Jay St|            40.7014851|           -73.98656928|           242|Carlton Ave & Flu...|           40.697787|           -73.973736|  25041|Subscriber|      1989|     1|
    |          335|2017-01-01 00:14:20|2017-01-01 00:19:55|            3429|Hanson Pl & Ashla...|     40.68506807308177|     -73.97790759801863|           270|Adelphi St & Myrt...|         40.69308257|         -73.97178913|  26119|Subscriber|      1990|     1|
    |          485|2017-01-01 00:15:08|2017-01-01 00:23:13|            3349|Grand Army Plaza ...|            40.6729679|           -73.97087984|          3414|Bergen St & Flatb...|  40.680944723477296|   -73.97567331790923|  17587|Subscriber|      1995|     2|
    |          486|2017-01-01 00:15:12|2017-01-01 00:23:19|            3349|Grand Army Plaza ...|            40.6729679|           -73.97087984|          3414|Bergen St & Flatb...|  40.680944723477296|   -73.97567331790923|  19764|Subscriber|      1963|     2|
    |          391|2017-01-01 00:15:17|2017-01-01 00:21:48|            3349|Grand Army Plaza ...|            40.6729679|           -73.97087984|          3417|   Baltic St & 5 Ave|          40.6795766|         -73.97854971|  18945|Subscriber|      1968|     2|
    |         1453|2017-01-01 00:17:13|2017-01-01 00:41:26|            3349|Grand Army Plaza ...|            40.6729679|           -73.97087984|           343|Clinton Ave & Flu...|            40.69794|         -73.96986848|  25744|  Customer|      null|     0|
    |         1471|2017-01-01 00:17:14|2017-01-01 00:41:46|            3349|Grand Army Plaza ...|            40.6729679|           -73.97087984|           343|Clinton Ave & Flu...|            40.69794|         -73.96986848|  27273|  Customer|      null|     0|
    |          966|2017-01-01 00:17:27|2017-01-01 00:33:34|            3319|       14 St & 5 Ave|             40.666287|           -73.98895053|           364|Lafayette Ave & C...|         40.68900443|         -73.96023854|  25525|Subscriber|      1990|     1|
    |          301|2017-01-01 00:17:50|2017-01-01 00:22:52|             389| Broadway & Berry St|           40.71044554|           -73.96525063|          3073|Division Ave & Ho...|         40.70691254|         -73.95441667|  18369|Subscriber|      1983|     2|
    |          184|2017-01-01 00:18:46|2017-01-01 00:21:50|            3064|Myrtle Ave & Lewi...|           40.69681963|           -73.93756926|          3059|Pulaski St & Marc...|          40.6933982|           -73.939877|  19078|Subscriber|      1984|     1|
    +-------------+-------------------+-------------------+----------------+--------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    only showing top 10 rows
    
    Cluster4
    +-------------+-------------------+-------------------+----------------+------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |Trip Duration|         Start Time|          Stop Time|Start Station ID|Start Station Name|Start Station Latitude|Start Station Longitude|End Station ID|    End Station Name|End Station Latitude|End Station Longitude|Bike ID| User Type|Birth Year|Gender|
    +-------------+-------------------+-------------------+----------------+------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    |          826|2017-01-01 00:05:37|2017-01-01 00:19:24|             362|Broadway & W 37 St|           40.75172632|           -73.98753523|           445|  E 10 St & Avenue A|         40.72740794|         -73.98142006|  23288|Subscriber|      1977|     2|
    |         1081|2017-01-01 00:07:49|2017-01-01 00:25:50|             528|   2 Ave & E 31 St|           40.74290902|           -73.97706058|           526|     E 33 St & 5 Ave|         40.74765947|         -73.98490707|  26138|Subscriber|      1993|     1|
    |          479|2017-01-01 00:08:00|2017-01-01 00:15:59|             474|   5 Ave & E 29 St|            40.7451677|           -73.98683077|          3259|     9 Ave & W 28 St|   40.74937024193277|   -73.99923384189606|  19728|Subscriber|      1973|     1|
    |         2005|2017-01-01 00:05:57|2017-01-01 00:39:23|             524|   W 43 St & 6 Ave|           40.75527307|           -73.98316936|          3325|     E 95 St & 3 Ave|          40.7849032|           -73.950503|  17171|Subscriber|      1992|     1|
    |          738|2017-01-01 00:08:50|2017-01-01 00:21:09|             297|   E 15 St & 3 Ave|             40.734232|             -73.986923|           355|Bayard St & Baxte...|         40.71602118|         -73.99974372|  23874|Subscriber|      1996|     1|
    |          901|2017-01-01 00:09:18|2017-01-01 00:24:20|             515|  W 43 St & 10 Ave|           40.76009437|           -73.99461843|          3428|     8 Ave & W 16 St|           40.740983|           -74.001702|  26501|Subscriber|      1964|     1|
    |          899|2017-01-01 00:09:20|2017-01-01 00:24:19|             515|  W 43 St & 10 Ave|           40.76009437|           -73.99461843|          3428|     8 Ave & W 16 St|           40.740983|           -74.001702|  15597|Subscriber|      1970|     1|
    |         1504|2017-01-01 00:09:29|2017-01-01 00:34:34|             423|   W 54 St & 9 Ave|           40.76584941|           -73.98690506|          3263|Cooper Square & E...|   40.72923649910006|   -73.99086803197861|  17810|Subscriber|      1994|     1|
    |          759|2017-01-01 00:09:30|2017-01-01 00:22:09|             267|Broadway & W 36 St|           40.75097711|           -73.98765428|           520|     W 52 St & 5 Ave|         40.75992262|         -73.97648516|  27279|  Customer|      null|     0|
    |         6370|2017-01-01 00:11:34|2017-01-01 01:57:44|             449|   W 52 St & 9 Ave|           40.76461837|           -73.98789473|           524|     W 43 St & 6 Ave|         40.75527307|         -73.98316936|  15603|  Customer|      null|     0|
    +-------------+-------------------+-------------------+----------------+------------------+----------------------+-----------------------+--------------+--------------------+--------------------+---------------------+-------+----------+----------+------+
    only showing top 10 rows
    



```python
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix

save1 = prediction.select("Start Station ID", "cluster","Start Station Latitude", "Start Station Longitude")
ts = save1.toPandas()
ts.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start Station ID</th>
      <th>cluster</th>
      <th>Start Station Latitude</th>
      <th>Start Station Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3226</td>
      <td>0</td>
      <td>40.782750</td>
      <td>-73.971370</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3263</td>
      <td>2</td>
      <td>40.729236</td>
      <td>-73.990868</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3143</td>
      <td>0</td>
      <td>40.776829</td>
      <td>-73.963888</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3143</td>
      <td>0</td>
      <td>40.776829</td>
      <td>-73.963888</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3143</td>
      <td>0</td>
      <td>40.776829</td>
      <td>-73.963888</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = ts
fig, ax = plt.subplots(1, 1, figsize=(6, 12))

def scatter(group):
    plt.plot(group['Start Station Longitude'],
             group['Start Station Latitude'],
             'o', label=group.name)

df.groupby('cluster').apply(scatter)



ax.set(xlabel='Start Station Longitude',
       ylabel='Start Station Latitude',
       title='Locations',
       xlim=[-74.04,-73.91],
       ylim=[40.62,40.82])

ax.legend(loc=2)
```




    <matplotlib.legend.Legend at 0x7fd5baa9ee90>




![png](output_10_1.png)



```python
save2 = prediction.groupBy("cluster").count().orderBy("cluster")
ts2 = save2.toPandas()
ts2.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>414333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>633549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>319574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>878530</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
