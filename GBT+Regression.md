
# GBT Regression


### Load the Data
We begin by loading our data, which is stored in csv format. We cache the data so that we only read it from disk once.


```python
# We use the sqlContext.read method to read the data and set a few options:
#  'format': specifies the Spark CSV data source
#  'header': set to true to indicate that the first line of the CSV data file is a header
# The file is called 'hour.csv'.
if sc.version >= '2.0':
  # Spark 2.0+ includes CSV as a native Spark SQL datasource.
  citibike = sqlContext.read.format('csv').option("header", 'true').load("/FileStore/tables/93pjcoa21493774548925/citibike2017.csv")
else:
  # Earlier Spark versions can use the Spark CSV package
  citibike = sqlContext.read.format('com.databricks.spark.csv').option("header", 'true').load("/FileStore/tables/93pjcoa21493774548925/citibike2017.csv")
# Calling cache on the DataFrame will make sure we persist it in memory the first time it is used.
# The following uses will be able to read from memory, instead of re-reading the data from disk.
citibike.cache()
```

### Display the Data
Call the display function to see a sample of the data


```python
display(citibike)
```


```python
print "Our dataset has %d rows." % citibike.count()
```

Print the schema of our dataset to see the type of each column.


```python
citibike.printSchema()
```

The DataFrame is currently using strings, but we know all columns are numeric. Let's cast them.


```python
# The following call takes all columns (df.columns) and casts them using Spark SQL to a numeric type (DoubleType).
from pyspark.sql.functions import col  # for indicating a column using a string in the line below
citibike = citibike.select([col(c).cast("double").alias(c) for c in citibike.columns])
citibike.printSchema()
```

### Split the Data
You will use 70% of the data for training and reserve 30% for testing. In the testing data, the label column in renamed to trueLabel so you can use it later to compare predicted labels with known actual values.


```python
# Split the dataset randomly into 70% for training and 30% for testing.
train, test = citibike.randomSplit([0.7, 0.3])
print "We have %d training examples and %d test examples." % (train.count(), test.count())
```

### Visualize the data


```python
display(train.select("StartDate", "TripDuration"))
```

### Train a Pipeline
First, we define the feature processing stages of the Pipeline:

[1] Assemble feature columns into a feature vector   [2] Identify categorical features, and index them.


```python
from pyspark.ml.feature import VectorAssembler, VectorIndexer
featuresCols = citibike.columns
featuresCols.remove('TripDuration')
# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)
```

Second, we define the model training stage of the Pipeline.

GBTRegressor takes feature vectors and labels as input and learns to predict labels of new examples.


```python
from pyspark.ml.regression import GBTRegressor
# Takes the "features" column and learns to predict "cnt"
gbt = GBTRegressor(labelCol="TripDuration")
```

Third, we wrap the model training stage within a CrossValidator stage. 

CrossValidator knows how to call the GBT algorithm with different hyperparameter settings. It will train multiple models and choose the best one, based on minimizing some metric, which is Root Mean Squared Error (RMSE).


```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
# Define a grid of hyperparameters to test:
#  - maxDepth: max depth of each decision tree in the GBT ensemble
#  - maxIter: iterations, i.e., number of trees in each GBT ensemble
# In this example notebook, we keep these values small.  In practice, to get the highest accuracy, you would likely want to try deeper trees (10 or higher) and more trees in the ensemble (>100).
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
# Declare the CrossValidator, which runs model tuning for us.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)
```

Finally, we can tie our feature processing and model training stages together into a single Pipeline.


```python
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])
```

### Train the Pipeline


```python
pipelineModel = pipeline.fit(train)
```

### Make predictions and Evaluate results
Our final step will be to use our fitted model to make predictions on new data.

We also evaluate our predictions.


```python
predictions = pipelineModel.transform(test)
```


```python
display(predictions.select("TripDuration", "prediction", *featuresCols))
```

### Root Mean Square Error (RMSE)
The RMSE indicates the average seconds between predicted and actual trip duration


```python
rmse = evaluator.evaluate(predictions)
print "RMSE on our test set: %g" % rmse
```

# Result shows:
### Root Mean Square Error (RMSE): 3369.09

### Visualization: 
Plotting predictions vs TripDuration


```python
display(predictions.select("TripDuration", "prediction"))
```
