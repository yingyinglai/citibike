
# Linear Regression

### Prepare the Data

First, import the libraries you will need and prepare the training and test data


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
```

### Load Source Data
The data for this project is provided as a CSV containing details of Citi Bike Company.
Import csv and create table. Name the table as citibiker


Then, read csv from table and show 5 rows of the table.


```python
  bikeSchema = StructType([
  StructField("TripDuration", IntegerType(), False),
  StructField("StartDate", IntegerType(), False),
  StructField("Startampm", IntegerType(), False),
  StructField("Endampm", IntegerType(), False),
  StructField("UserType", IntegerType(), False),
  StructField("Gender", IntegerType(), False),
])
```


```python
data_csv = spark.sql ("SELECT * FROM citibiker")

data_csv.show(5)
```

### Prepare the Data
Select a subset of columns to use as features and create a label.

### Split the Data
You will use 70% of the data for training and reserve 30% for testing. In the testing data, the label column in renamed to trueLabel so you can use it later to compare predicted labels with known actual values.


```python
# Select features and label
data = data_csv.select( "StartDate", "Startampm", "Endampm", "UserType", "Gender", col("TripDuration").alias("label"))

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
```

### Pipeline
Define the pipeline that creates a feature vector with maxIter=10 and regParam=0.3, then train the regression model


```python
# Define the pipeline
assembler = VectorAssembler(inputCols = ["StartDate", "Startampm", "Endampm", "UserType", "Gender"], outputCol="features")
lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)
pipeline = Pipeline(stages=[assembler, lr])

# Train the model
piplineModel = pipeline.fit(train)
```

### Test the model
apply the model to the test data


```python
prediction = piplineModel.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show()
```

### Examine the Predicted and Actual Values


```python
predicted.createOrReplaceTempView("regressionPredictions")
```


```python
# Reference: http://standarderror.github.io/notes/Plotting-with-PySpark/
dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")

display(dataPred)
```

### Root Mean Square Error (RMSE)
Use Regression Evaluator class to retrieve the RMSE. The RMSE indicates the average seconds between predicted and actual trip duration


```python
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print "Root Mean Square Error (RMSE):", rmse
```

# Result shows:
### Root Mean Square Error (RMSE): 4928.843703
