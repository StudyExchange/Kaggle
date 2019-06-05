import json


def save_predict_result(pred_result, redis_pred_result_name, redis_cli):
    pred_result_ser = json.dumps(pred_result)
    print(redis_cli.rpush(redis_pred_result_name, pred_result_ser))


def get_predict_result(redis_pred_result_name, redis_cli, nrows, skip=0):
    pred_result_ser_arr = redis_cli.lrange(
        redis_pred_result_name, skip, skip+nrows)
    pred_result_arr = [json.loads(
        pred_result_ser) for pred_result_ser in pred_result_ser_arr if pred_result_ser]
    return pred_result_arr
