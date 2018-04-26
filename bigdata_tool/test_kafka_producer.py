from kafka import KafkaProducer
import datetime,time
producer = KafkaProducer(bootstrap_servers=["172.16.36.202:21005","172.16.36.201:21005"],client_id='aograph_wmo')


while True:
    v = str(datetime.datetime.now()).encode("utf-8")
    print(v)
    producer.send(topic="aograph_tes1",value=v)
    producer.flush()
    time.sleep(3)