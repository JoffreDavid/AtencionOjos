from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, window

spark = SparkSession.builder.appName("AttentionAnalyzer").getOrCreate()
df = spark.read.json("/app/data/attention_logs.json")

df = df.withColumn("timestamp", df["timestamp"].cast("timestamp"))

result = df.groupBy("usuario", window("timestamp", "1 minute")) \
           .agg(avg("nivel_atencion").alias("promedio"))

result.show()
result.coalesce(1).write.mode("overwrite").json("/app/data/resultados")