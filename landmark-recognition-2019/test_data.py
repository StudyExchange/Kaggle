import os
import json
import redis

from config import Config


def init_test_idx_arr(test_idx_start, test_idx_end, test_idx_arr_name, redis_cli):
    test_idx_arr = list(range(test_idx_start, test_idx_end))
    # for test_idx in test_idx_arr:
    #     print(type(test_idx), test_idx)
    redis_cli.delete(test_idx_arr_name)
    print(redis_cli.lpush(test_idx_arr_name, *test_idx_arr))


def get_one_test_image_index(test_idx_arr_name, redis_cli):
    test_idx_ser = redis_cli.rpop(test_idx_arr_name)
    if test_idx_ser:
        return json.loads(test_idx_ser)


if __name__ == '__main__':
    # Init redis connection
    redis_cli = redis.Redis(
        host=Config.HOST, port=Config.PORT, password=Config.PASSWORD)
    # There can set task data manually
    init_test_idx_arr(0, 117577, Config.REDIS_TEST_IDX_ARR_NAME, redis_cli) # test2 data: 117577
