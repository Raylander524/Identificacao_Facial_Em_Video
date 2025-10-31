import numpy as np
from flask import Flask, request, render_template, url_for, jsonify,send_from_directory
import os
import cv2
from ultralytics import YOLO
import insightface
from werkzeug.utils import secure_filename
import threading
import queue
import base64
import faiss
import pickle
import uuid
import shutil
import onnxruntime as ort

# === Configuração Flask ===
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/resultados"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
PASTA_INDICE = "index"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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

def carregar_indices():
    indice_path = os.path.join(PASTA_INDICE, "indice_rostos.index")
    nomes_path = os.path.join(PASTA_INDICE, "nomes.pkl")
    
    if os.path.exists(indice_path) and os.path.exists(nomes_path):
        index = faiss.read_index(indice_path)
        with open(nomes_path, "rb") as f:
            nomes = pickle.load(f)
        return index, nomes
    
    return None, None

indexes,nomes = carregar_indices()


def processar_frame(frame):
    resultados = []
    resultado_indices = []
    yolo_result = yolo.predict(frame, conf=0.5, classes=[0])
    for box in yolo_result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        faces = model.get(face_crop)
        if len(faces) > 0:
            embedding = faces[0].embedding.reshape(1, -1).astype('float32')
            emb_norm = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            distances, indices = indexes.search(emb_norm, k=1)

            for i, dist in enumerate(distances[0]):
                print(f"Match {i+1}: distância = {dist:.4f}, nome = {nomes[indices[0][i]]}")
            if dist <= 1.0:
                resultado_indices.append(indices[0][i])

            if resultado_indices:
                result = [nomes[idx] for idx in resultado_indices]
                resultados.append((result[0], face_crop))
            
    return resultados

def processar_video(video_path, batch_size=15):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imagens_detectadas = {}
    frame_count = 0
    progresso["percent"] = 0

    frames_batch = []
    frame_indices = []

    frame_skip = 10  # processa 1 a cada 10 frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # frame = cv2.resize(frame, (640, 360))
        frames_batch.append(frame)
        frame_indices.append(frame_count)
        frame_count += 1
        resultado_indices = []
        # Quando atingir o tamanho do lote ou acabar o vídeo
        if len(frames_batch) == batch_size or not cap.isOpened():
            # Processa o lote de frames com YOLO
            yolo_results = yolo.predict(frames_batch, conf=0.5, classes=[0])
            for idx, yolo_result in enumerate(yolo_results):
                frame = frames_batch[idx]
                for box in yolo_result.boxes:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_crop = frame[y1:y2, x1:x2]
                    faces = model.get(face_crop)

                    if len(faces) > 0:
                        embedding = faces[0].embedding.reshape(1, -1).astype('float32')
                        emb_norm = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                        distances, indices = indexes.search(emb_norm, k=1)
                        
                        for i, dist in enumerate(distances[0]):
                            print(f"Match {i+1}: distância = {dist:.4f}, nome = {nomes[indices[0][i]]}")
                        if dist <= 1.0:
                            resultado_indices.append(indices[0][i])
                            nome_img =uuid.uuid4().hex
                            recorte_nome = f"{nome_img}_frame_{frame_indices[idx]}.jpg"
                            recorte_path = os.path.join(RESULT_FOLDER, recorte_nome)
                            cv2.imwrite(recorte_path, face_crop)

                        result = [nomes[idx] for idx in resultado_indices]

                        imagens_detectadas[nome_img] = {
                                    "referencia": result[0],
                                    "recorte": recorte_path
                                }

        progresso["percent"] = int((frame_count / total_frames) * 100)

    cap.release()
    progresso["percent"] = 100
    return imagens_detectadas

# Fila para frames recebidos
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker_identificacao():
    pessoas_encontradas = set()
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
    quantidade = int(request.form.get("quantidade", 5))
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    frame = cv2.imread(filepath)
    if frame is None:
        return jsonify({"mensagem": "Erro ao ler a imagem enviada.", "encontrado": False})

    faces = model.get(frame)
    if len(faces) == 0:
        return render_template("resultado_foto.html", resultados=[])

    resultados = []
    for face in faces:
        embedding = face.embedding.reshape(1, -1).astype("float32")
        emb_norm = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        if indexes is None or nomes is None:
            return render_template("resultado_foto.html", resultados=[])

        distances, indices = indexes.search(emb_norm, k=quantidade)

        for i, dist in enumerate(distances[0]):
            if dist <= 1.0:
                nome_ref = nomes[indices[0][i]]
                # Recorte do rosto em base64
                x1, y1, x2, y2 = map(int, face.bbox)
                face_crop = frame[y1:y2, x1:x2]
                _, buffer = cv2.imencode(".jpg", face_crop)
                img_base64 = base64.b64encode(buffer).decode("utf-8")

                resultados.append({
                    "nome": nome_ref,
                    "distancia": float(dist),
                    "foto": img_base64
                })

    return render_template("resultado_foto.html", resultados=resultados)


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
    for nome_img, face_crop in resultados:
        # Foto recortada do vídeo
        _, buffer = cv2.imencode('.jpg', face_crop)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        # Foto arquivada (referência)
        ref_img = cv2.imread(nome_img)
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
    return send_from_directory('/app/rostos_dataset', filename)

# rota pra servir arquivos de novas_imagens
@app.route('/novas_imagens/<path:filename>')
def serve_novas_imagens(filename):
    return send_from_directory('/app/novas_imagens', filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context=('/app/cert.pem','/app/key.pem'))



