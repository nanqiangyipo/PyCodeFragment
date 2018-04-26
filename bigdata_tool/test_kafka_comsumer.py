from kafka import KafkaConsumer

# consumer = KafkaConsumer('aograph_tes1',bootstrap_servers=["172.16.36.202:21005","172.16.36.201:21005"])
consumer = KafkaConsumer('aograph-android-mmi',bootstrap_servers=["172.16.36.202:21005","172.16.36.201:21005"])

for msg in consumer:
    # print(msg.value.decode(), msg.timestamp)
    print(msg.timestamp,msg.value.decode() )