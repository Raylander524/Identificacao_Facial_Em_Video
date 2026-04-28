import numpy as np
import requests
from flask import Flask, request, render_template, url_for, jsonify,send_from_directory,Response
import os
import cv2
from ultralytics import YOLO
import insightface
from werkzeug.utils import secure_filename
import threading
import queue
import base64
import uuid
import shutil
import time
import onnxruntime as ort
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

# === Configuração Flask ===
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/resultados"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
PASTA_INDICE = "index"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
OCR_API_URL = "http://200.129.44.250:9000/process"   # pode ser "http://localhost:9000" se for rodar local local
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rostos_embeddings")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))
MILVUS_COSINE_THRESHOLD = 0.5
ROSTOS_DATASET_DIR = os.getenv("ROSTOS_DATASET_DIR", "")
IMAGENS_ADICIONADAS_DIR = os.getenv("IMAGENS_ADICIONADAS_DIR", "imagens_adicionadas")
os.makedirs(IMAGENS_ADICIONADAS_DIR, exist_ok=True)
RTSP_URL = "rtsp://teste:abc123456@200.19.182.251:8080/cam/realmonitor?channel=1&subtype=0"
frame_lock = threading.Lock()
cap_rtsp = None
latest_frame = None
latest_jpeg = None

# === Carregar modelos ===
yolo = YOLO("yolo11n.pt")
available_providers = ort.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = 0
else:
    providers = ["CPUExecutionProvider"]
    ctx_id = -1
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
model.prepare(ctx_id=0, det_size=(320, 320))

progresso = {"percent": 0}


def obter_diretorio_dataset():
    candidatos = [ROSTOS_DATASET_DIR, "rostos_dataset", "/rostos_dataset"]
    for caminho in candidatos:
        if caminho and os.path.isdir(caminho):
            return caminho
    return None


def referencia_para_url(caminho):
    caminho_norm = os.path.abspath(caminho)
    dataset_dir = obter_diretorio_dataset()
    if dataset_dir:
        dataset_abs = os.path.abspath(dataset_dir)
        if caminho_norm.startswith(dataset_abs + os.sep):
            relativo = os.path.relpath(caminho_norm, dataset_abs).replace("\\", "/")
            return f"/rostos_dataset/{relativo}"

    imagens_add_abs = os.path.abspath(IMAGENS_ADICIONADAS_DIR)
    if caminho_norm.startswith(imagens_add_abs + os.sep):
        relativo = os.path.relpath(caminho_norm, imagens_add_abs).replace("\\", "/")
        return f"/imagens_adicionadas/{relativo}"

    return caminho


def referencia_para_arquivo(referencia):
    if referencia.startswith("/imagens_adicionadas/"):
        relativo = referencia.replace("/imagens_adicionadas/", "", 1)
        return os.path.join(IMAGENS_ADICIONADAS_DIR, relativo)

    if referencia.startswith("/rostos_dataset/"):
        dataset_dir = obter_diretorio_dataset()
        if dataset_dir:
            relativo = referencia.replace("/rostos_dataset/", "", 1)
            return os.path.join(dataset_dir, relativo)

    return referencia


def listar_imagens_dataset(dataset_dir):
    extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for raiz, _, arquivos in os.walk(dataset_dir):
        for arquivo in arquivos:
            if os.path.splitext(arquivo)[1].lower() in extensoes:
                yield os.path.join(raiz, arquivo)


def extrair_embedding_principal(imagem):
    faces = model.get(imagem)
    if len(faces) == 0:
        return None

    # Escolhe o maior rosto quando há múltiplas detecções na imagem.
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    embedding = face.embedding.astype("float32")
    norma = np.linalg.norm(embedding)
    if norma == 0:
        return None
    return embedding / norma


def popular_collection_dataset(collection, dataset_dir, batch_size=128):
    referencias_batch = []
    embeddings_batch = []
    total_imagens = 0
    total_inseridos = 0

    for caminho_img in listar_imagens_dataset(dataset_dir):
        total_imagens += 1
        imagem = cv2.imread(caminho_img)
        if imagem is None:
            continue

        embedding = extrair_embedding_principal(imagem)
        if embedding is None:
            continue

        referencias_batch.append(referencia_para_url(caminho_img))
        embeddings_batch.append(embedding.tolist())

        if len(referencias_batch) >= batch_size:
            collection.insert([referencias_batch, embeddings_batch])
            total_inseridos += len(referencias_batch)
            referencias_batch.clear()
            embeddings_batch.clear()

    if referencias_batch:
        collection.insert([referencias_batch, embeddings_batch])
        total_inseridos += len(referencias_batch)

    if total_inseridos > 0:
        collection.flush()

    print(
        f"Carga inicial do Milvus concluída: {total_inseridos} embeddings "
        f"a partir de {total_imagens} imagens do dataset."
    )

def carregar_collection_milvus():
    ultima_excecao = None
    for tentativa in range(15):
        try:
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
            break
        except Exception as e:
            ultima_excecao = e
            print(f"Milvus indisponível (tentativa {tentativa + 1}/15): {e}")
            time.sleep(2)
    else:
        print(f"Falha ao conectar no Milvus: {ultima_excecao}")
        return None

    colecao_criada = False
    if not utility.has_collection(MILVUS_COLLECTION):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="nome", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields=fields, description="Embeddings faciais")
        collection = Collection(name=MILVUS_COLLECTION, schema=schema)
        colecao_criada = True
    else:
        collection = Collection(name=MILVUS_COLLECTION)

    if colecao_criada:
        dataset_dir = obter_diretorio_dataset()
        if dataset_dir is None:
            print(
                "Coleção Milvus criada, mas o diretório rostos_dataset não foi encontrado."
            )
        else:
            print(f"Coleção Milvus não existia. Carregando dataset de: {dataset_dir}")
            popular_collection_dataset(collection, dataset_dir)

    if len(collection.indexes) == 0:
        try:
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                },
            )
        except Exception as e:
            print(f"Não foi possível criar índice IVF_FLAT/COSINE: {e}")

    collection.load()
    return collection


def buscar_similares(collection, embedding, k=1):
    if collection is None:
        return []

    embedding_norm = embedding / np.linalg.norm(embedding)
    resultados = collection.search(
        data=[embedding_norm.tolist()],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=k,
        output_fields=["nome"],
    )

    similares = []
    for hit in resultados[0]:
        similaridade = float(hit.distance)
        if similaridade > MILVUS_COSINE_THRESHOLD:
            similares.append({
                "nome": hit.entity.get("nome"),
                "similaridade": similaridade,
            })

    return similares


def inserir_referencia_no_milvus(collection, referencia, embedding):
    if collection is None:
        return False

    embedding_norm = embedding / np.linalg.norm(embedding)
    collection.insert([[referencia], [embedding_norm.tolist()]])
    collection.flush()
    return True


collection_rostos = carregar_collection_milvus()


def processar_frame(frame):
    resultados = []
    yolo_result = yolo.predict(frame, conf=0.5, classes=[0])
    for box in yolo_result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        faces = model.get(face_crop)
        if len(faces) > 0:
            embedding = faces[0].embedding.astype("float32")
            similares = buscar_similares(collection_rostos, embedding, k=1)

            for i, similar in enumerate(similares):
                print(
                    f"Match {i+1}: similaridade = {similar['similaridade']:.4f}, "
                    f"nome = {similar['nome']}"
                )

            if similares:
                referencia_url = similares[0]["nome"]
                referencia_arquivo = referencia_para_arquivo(referencia_url)
                resultados.append((similares[0]["nome"], referencia_arquivo, face_crop))
            
    return resultados

def processar_video(video_path, batch_size=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    imagens_detectadas = {}
    frame_count = 0
    progresso["percent"] = 0

    frames_batch = []
    frame_indices = []

    frame_skip = 15  # processa 1 a cada 10 frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, (800, 450))
        frames_batch.append(frame)
        frame_indices.append(frame_count)
        frame_count += 1

        # Processa o batch assim que atingir o tamanho máximo
        if len(frames_batch) == batch_size:
            print(f"Processando lote com {len(frames_batch)} frames...")
            processar_lote(frames_batch, frame_indices, imagens_detectadas)
            frames_batch.clear()
            frame_indices.clear()

        progresso["percent"] = int((frame_count / total_frames) * 100)

    if len(frames_batch) > 0:
        processar_lote(frames_batch, frame_indices, imagens_detectadas)

    cap.release()
    progresso["percent"] = 100
    return imagens_detectadas


def processar_lote(frames_batch, frame_indices, imagens_detectadas):
    yolo_results = yolo.predict(frames_batch, conf=0.5, classes=[0])

    for idx, yolo_result in enumerate(yolo_results):
        frame = frames_batch[idx]
        print(f"Frame {frame_indices[idx]} -> {len(yolo_result.boxes)} detecções")

        for box in yolo_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]
            faces = model.get(face_crop)

            if len(faces) == 0:
                continue

            embedding = faces[0].embedding.astype("float32")
            similares = buscar_similares(collection_rostos, embedding, k=1)

            for i, similar in enumerate(similares):
                print(
                    f"Match {i+1}: similaridade = {similar['similaridade']:.4f}, "
                    f"nome = {similar['nome']}"
                )

                if similar["similaridade"] > MILVUS_COSINE_THRESHOLD:
                    nome_img = uuid.uuid4().hex
                    recorte_nome = f"{nome_img}frame{frame_indices[idx]}.jpg"
                    recorte_path = os.path.join(RESULT_FOLDER, recorte_nome)
                    cv2.imwrite(recorte_path, face_crop)

                    imagens_detectadas[nome_img] = {
                        "referencia": similar["nome"],
                        "recorte": recorte_path,
                    }

def inicializar_rtsp():
    global cap_rtsp
    cap_rtsp = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap_rtsp.isOpened():
        raise RuntimeError("Não foi possível abrir o RTSP")

def rtsp_reader():
    global cap_rtsp, latest_frame, latest_jpeg
    while True:
        try:
            if cap_rtsp is None or not cap_rtsp.isOpened():
                try:
                    inicializar_rtsp()
                except Exception as e:
                    time.sleep(2)
                    continue

            ret, frame = cap_rtsp.read()
            if not ret:
                # tenta reabrir
                cap_rtsp.release()
                cap_rtsp = None
                time.sleep(0.5)
                continue

            ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                continue

            with frame_lock:
                latest_frame = frame
                latest_jpeg = buffer.tobytes()

            # controle simples de taxa (~10 fps)
            # time.sleep(0.1)
        except Exception:
            time.sleep(0.5)

def generate_dvr_stream():
    global latest_jpeg
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    while True:
        with frame_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is None:
            time.sleep(0.1)
            continue
        yield boundary + frame_bytes + b'\r\n'

# Fila para frames recebidos
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker_identificacao():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break  # Para a thread
        resultados = processar_frame(frame)
        if resultados:
            nome_img, referencia, face_crop = resultados[0]
            mensagem = f"Pessoa reconhecida: {nome_img}"
            encontrado = True
        else:
            mensagem = "Nenhuma pessoa reconhecida."
            encontrado = False
        result_queue.put({"mensagem": mensagem, "encontrado": encontrado})

# Inicie a thread ao iniciar o app
threading.Thread(target=worker_identificacao, daemon=True).start()
threading.Thread(target=rtsp_reader, daemon=True).start()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Recebe vídeo da webcam ou upload
        if "video" in request.files:
            file = request.files["video"]
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Processa o vídeo normalmente
            imagens_detectadas = processar_video(filepath)

            # copia imagens para pasta static/resultados
            imagens_para_exibir = []
            for chave, dados in imagens_detectadas.items():
                # Copia recorte também
                destino_rec = os.path.join(RESULT_FOLDER, os.path.basename(dados["recorte"]))
                if not os.path.exists(destino_rec):
                    shutil.copy(dados["recorte"], destino_rec)

                # Adiciona caminhos para exibição
                imagens_para_exibir.append({
                    "referencia": dados['referencia'],
                    "recorte": url_for("static", filename=f"resultados/{os.path.basename(dados['recorte'])}")
                })

            return render_template("resultado.html", imagens=imagens_para_exibir)

    return render_template("index.html")

@app.route("/progresso")
def get_progresso():
    return jsonify(progresso)

@app.route("/analisar_foto", methods=["POST"])
def analisar_foto():
    if "foto" not in request.files:
        return jsonify({"mensagem": "Nenhuma foto enviada.", "encontrado": False})

    file = request.files["foto"]
    try:
        quantidade = int(request.form.get("quantidade", 5))
    except (TypeError, ValueError):
        quantidade = 5
    quantidade = max(1, quantidade)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    frame = cv2.imread(filepath)
    if frame is None:
        return jsonify({"mensagem": "Erro ao ler a imagem enviada.", "encontrado": False})

    faces = model.get(frame)
    if len(faces) == 0:
        return render_template("resultado_foto.html", resultados=[])

    # Usa o maior rosto da imagem de consulta para buscar os mais similares.
    face_consulta = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    resultados = []
    if collection_rostos is None:
        return render_template("resultado_foto.html", resultados=[])

    embedding = face_consulta.embedding.astype("float32")
    similares = buscar_similares(collection_rostos, embedding, k=quantidade)

    # Recorte do rosto de consulta em base64
    x1, y1, x2, y2 = map(int, face_consulta.bbox)
    face_crop = frame[y1:y2, x1:x2]
    _, buffer = cv2.imencode(".jpg", face_crop)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    for similar in similares[:quantidade]:
        resultados.append({
            "nome": similar["nome"],
            "distancia": float(similar["similaridade"]),
            "foto": img_base64
        })

    return render_template("resultado_foto.html", resultados=resultados)


@app.route("/adicionar_referencia", methods=["POST"])
def adicionar_referencia():
    if "foto" not in request.files:
        return jsonify({"mensagem": "Nenhuma foto enviada.", "adicionada": False}), 400

    if collection_rostos is None:
        return jsonify({"mensagem": "Milvus indisponível.", "adicionada": False}), 503

    file = request.files["foto"]
    if file.filename == "":
        return jsonify({"mensagem": "Arquivo inválido.", "adicionada": False}), 400

    extensao = os.path.splitext(secure_filename(file.filename))[1].lower() or ".jpg"
    nome_arquivo = f"{uuid.uuid4().hex}{extensao}"
    caminho_salvo = os.path.join(IMAGENS_ADICIONADAS_DIR, nome_arquivo)
    file.save(caminho_salvo)

    imagem = cv2.imread(caminho_salvo)
    if imagem is None:
        os.remove(caminho_salvo)
        return jsonify({"mensagem": "Erro ao ler a imagem enviada.", "adicionada": False}), 400

    embedding = extrair_embedding_principal(imagem)
    if embedding is None:
        os.remove(caminho_salvo)
        return jsonify({"mensagem": "Nenhum rosto detectado na imagem.", "adicionada": False}), 400

    referencia_url = referencia_para_url(caminho_salvo)
    inserido = inserir_referencia_no_milvus(collection_rostos, referencia_url, embedding)
    if not inserido:
        os.remove(caminho_salvo)
        return jsonify({"mensagem": "Falha ao inserir no Milvus.", "adicionada": False}), 500

    return jsonify({
        "mensagem": "Imagem adicionada e indexada com sucesso.",
        "adicionada": True,
        "referencia": referencia_url,
    })


@app.route("/processar_frame", methods=["POST"])
def processar_frame_webcam():
    if "frame" not in request.files:
        return jsonify({"mensagem": "Nenhum frame recebido.", "encontrado": False, "pessoas": []})
    file = request.files["frame"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    frame = cv2.imread(filepath)
    resultados = processar_frame(frame)
    pessoas = []
    for nome_img, referencia, face_crop in resultados:
        # Foto recortada do vídeo
        _, buffer = cv2.imencode('.jpg', face_crop)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        # Foto arquivada (referência)
        ref_img = cv2.imread(referencia)
        if ref_img is None:
            continue
        _, ref_buffer = cv2.imencode('.jpg', ref_img)
        ref_base64 = base64.b64encode(ref_buffer).decode('utf-8')
        pessoas.append({
            "nome": nome_img,
            "foto": img_base64,
            "foto_referencia": ref_base64
        })
    if pessoas:
        mensagem = f"{len(pessoas)} pessoa(s) reconhecida(s): " + ", ".join([p["nome"] for p in pessoas])
        encontrado = True
    else:
        mensagem = "Nenhuma pessoa reconhecida."
        encontrado = False
    return jsonify({
        "mensagem": mensagem,
        "encontrado": encontrado,
        "pessoas": pessoas
    })

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

# rota pra servir arquivos de rostos_dataset
@app.route('/rostos_dataset/<path:filename>')
def serve_rostos_dataset(filename):
    dataset_dir = obter_diretorio_dataset()
    if dataset_dir is None:
        return jsonify({"error": "Diretório rostos_dataset não encontrado."}), 404
    return send_from_directory(dataset_dir, filename)

# rota pra servir arquivos de novas_imagens
@app.route('/novas_imagens/<path:filename>')
def serve_novas_imagens(filename):
    return send_from_directory('novas_imagens', filename)


@app.route('/imagens_adicionadas/<path:filename>')
def serve_imagens_adicionadas(filename):
    return send_from_directory(IMAGENS_ADICIONADAS_DIR, filename)

@app.route("/processar_frame_dvr", methods=["GET"])
def processar_frame_dvr():
    global cap_rtsp

    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        return jsonify({
            "mensagem": "Aguardando frames do DVR...",
            "encontrado": False,
            "pessoas": []
        })

    resultados = processar_frame(frame)

    pessoas = []
    for nome_img, referencia, face_crop in resultados:
        _, buffer = cv2.imencode('.jpg', face_crop)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        ref_img = cv2.imread(referencia)
        _, ref_buffer = cv2.imencode('.jpg', ref_img)
        ref_base64 = base64.b64encode(ref_buffer).decode("utf-8")

        pessoas.append({
            "nome": nome_img,
            "foto": img_base64,
            "foto_referencia": ref_base64
        })

    if pessoas:
        mensagem = f"{len(pessoas)} pessoa(s) reconhecida(s): " + ", ".join(p["nome"] for p in pessoas)
        encontrado = True
    else:
        mensagem = "Nenhuma pessoa reconhecida."
        encontrado = False

    return jsonify({
        "mensagem": mensagem,
        "encontrado": encontrado,
        "pessoas": pessoas
    })

@app.route("/dvr")
def dvr():
    return render_template("dvr.html")

@app.route("/dvr_stream")
def dvr_stream():
    return Response(generate_dvr_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/ocr")
def ocr():
    return render_template("ocr.html")

@app.route("/api/ocr", methods=["POST"])
def proxy_ocr():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    files = {"file": (file.filename, file.stream, file.content_type)}

    try:
        # requisição interna pro FastAPI
        resp = requests.post(OCR_API_URL, files=files)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context=('cert.pem','key.pem'))



