import redis
from rediscluster import StrictRedisCluster


# Redis
startup_nodes = [
                    {"host": "172.16.36.200", "port": "22400"},
                    {"host": "172.16.36.201", "port": "22400"},
                    {"host": "172.16.36.202", "port": "22400"},
                 ]

rc = StrictRedisCluster(startup_nodes=startup_nodes, decode_responses=True, max_connections=50)
# print(rc.keys('aograph-android-*'))
# print(rc.keys('aograph-android-*'))

# rc.lpush()

# rc.set("aograph_in_asdf", "v3")
# print(rc.get("aograph-android-None"))
# print(rc.get("foo"))
#
# print(rc.dbsize())
# for x in rc.keys('*'):
#     print(x)
# rc.delete('foo')
# ks = rc.keys('aograph-android-*')
# print(len(ks))
