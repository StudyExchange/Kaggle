import json
import redis
from config import Config


def init_task(task_amount, topn, model_date_str, libary_batch_amount, redis_cli):
    task_arr = []
    for i in range(task_amount):
        task_arr.append({
            'id': i,
            'per_process_gpu_memory_fraction': 0.9/task_amount,
            'topn': topn,
            'model_date_str': model_date_str,
            'libary_batch_amount': libary_batch_amount
        })
    task_ser_arr = [json.dumps(task) for task in task_arr]
    for task_ser in task_ser_arr:
        print(type(task_ser), task_ser)
    redis_cli.delete('task_arr')
    print(redis_cli.lpush('task_arr', *task_ser_arr))


def get_one_task(redis_cli):
    task_ser = redis_cli.rpop('task_arr')
    if task_ser:
        return json.loads(task_ser)


if __name__ == '__main__':
    # Init redis connection
    redis_cli = redis.Redis(
        host=Config.HOST, port=Config.PORT, password=Config.PASSWORD)
    # There can set task data manually
    init_task(5, 1000, '171023', 80, redis_cli)
