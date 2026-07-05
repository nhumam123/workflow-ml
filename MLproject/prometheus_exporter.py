from flask import Flask, request, jsonify, Response
import requests
import time
import psutil  # Untuk monitoring sistem
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
 
app = Flask(__name__)

MODEL_SERVICE_UP = Gauge(
    'mdl_model_service_up',
    'Model service status: 1 = up, 0 = down'
)

MODEL_HEALTH_LATENCY = Histogram(
    'mdl_model_health_check_duration_seconds',
    'Latency for model health check'
)

# Tambahan: jumlah prediksi per kelas
# Cocok untuk prediction drift sederhana
PREDICTION_COUNT = Counter(
    'mdl_model_prediction_total',
    'Total predictions by class',
    ['prediction_class']
)

# Tambahan: error rate
ERROR_COUNT = Counter(
    'mdl_http_errors_total',
    'Total HTTP Errors',
    ['error_type']
)

REQUEST_STATUS_COUNT = Counter(
    'mdl_http_response_status_total',
    'Total HTTP responses by status code',
    ['status_code']
)

MODEL_API_URL = "http://127.0.0.1:5005/invocations"

DUMMY_PAYLOAD = {
    "dataframe_split": {
        "columns": ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"],
        "data": [
        [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149000000.62]
        ]
    }
}


# Metrik untuk API model
REQUEST_COUNT = Counter('mdl_http_requests_total', 'Total HTTP Requests')  # Total request yang diterima
REQUEST_LATENCY = Histogram('mdl_http_request_duration_seconds', 'HTTP Request Latency')  # Waktu respons API
THROUGHPUT = Counter('mdl_http_requests_throughput', 'Total number of requests per second')  # Throughput
 
# Metrik untuk sistem
CPU_USAGE = Gauge('mdl_system_cpu_usage', 'CPU Usage Percentage')  # Penggunaan CPU
RAM_USAGE = Gauge('mdl_system_ram_usage', 'RAM Usage Percentage')  # Penggunaan RAM

def check_model_service():
    start_time = time.time()

    try:
        response = requests.post(
            MODEL_API_URL,
            json=DUMMY_PAYLOAD,
            timeout=3
        )

        duration = time.time() - start_time
        MODEL_HEALTH_LATENCY.observe(duration)

        if response.status_code == 200:
            MODEL_SERVICE_UP.set(1)
            return True, response.json()

        MODEL_SERVICE_UP.set(0)
        return False, {
            "status_code": response.status_code,
            "response": response.text
        }

    except requests.exceptions.Timeout:
        MODEL_SERVICE_UP.set(0)
        return False, {"error": "Model service timeout"}

    except requests.exceptions.ConnectionError:
        MODEL_SERVICE_UP.set(0)
        return False, {"error": "Model service connection error"}

    except Exception as e:
        MODEL_SERVICE_UP.set(0)
        return False, {"error": str(e)}
    
@app.route('/health/model', methods=['GET'])
def model_health():
    is_up, detail = check_model_service()

    if is_up:
        return jsonify({
            "status": "UP",
            "model_api": MODEL_API_URL,
            "detail": detail
        }), 200

    return jsonify({
        "status": "DOWN",
        "model_api": MODEL_API_URL,
        "detail": detail
    }), 503


# Endpoint untuk Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    # Update metrik sistem setiap kali /metrics diakses
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  # Ambil data CPU usage (persentase)
    RAM_USAGE.set(psutil.virtual_memory().percent)  # Ambil data RAM usage (persentase)
    check_model_service()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


 
# Endpoint untuk mengakses API model dan mencatat metrik
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()  # Tambah jumlah request
    THROUGHPUT.inc()  # Tambah throughput (request per detik)
 
    # Kirim request ke API model
    # api_url = "http://127.0.0.1:5005/invocations"
    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()
 
    try:
        response = requests.post(api_url, json=data)
        print(response.status_code)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)  # Catat latensi

        # Catat status code dari API model
        REQUEST_STATUS_COUNT.labels(
            status_code=str(response.status_code)
        ).inc()

        # Kalau response dari model error
        if response.status_code != 200:
            print("go to eror response")
            ERROR_COUNT.labels(error_type="model_api_error").inc()
            return jsonify({
                "error": "Model API returned error",
                "status_code": response.status_code,
                "detail": response.text
            }), response.status_code

        result = response.json()
        
        if "predictions" in result:
            predictions = result["predictions"]

            if isinstance(predictions, list):
                for pred in predictions:
                    PREDICTION_COUNT.labels(
                        prediction_class=str(pred)
                    ).inc()
        
        return jsonify(response.json())
 
    except Exception as e:
        return jsonify({"errorWOW": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)