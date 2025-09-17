from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import os
import cv2
from ultralytics import YOLO
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
import threading
import queue
import base64

# === Configuração Flask ===
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/resultados"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# === Carregar modelos ===
yolo = YOLO("yolo11n.pt")
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(320, 320))

# === Carregar imagens de referência ===
def buscar_imagens_recursivo(diretorio, extensoes={".jpg", ".jpeg", ".png"}):
    arquivos = []
    for root, _, files in os.walk(diretorio):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensoes:
                arquivos.append(os.path.join(root, file))
    return arquivos

imagens_referencia = buscar_imagens_recursivo("/mnt/Projeto_id_facial/PastaTeste/")  # <- pasta com seu banco
embedding_db = []
for img_path in imagens_referencia:
    img = cv2.imread(img_path)
    if img is None:
        continue
    faces_ref = model.get(img)
    for i, f in enumerate(faces_ref):
        embedding_db.append({
            "id": f"{os.path.basename(img_path)}_face_{i}",
            "embedding": f.embedding,
            "path": img_path
        })

progresso = {"percent": 0}

def processar_frame(frame, embedding_db, pessoas_encontradas):
    resultados = []
    yolo_result = yolo.predict(frame, conf=0.5, classes=[0], device="cuda")
    for box in yolo_result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        faces = model.get(face_crop)
        if len(faces) > 0:
            embedding = faces[0].embedding
            sims = [
                (ref["id"], cosine_similarity([ref["embedding"]], [embedding])[0][0], ref["path"])
                for ref in embedding_db
                if ref["id"].split("_face_")[0] not in pessoas_encontradas
            ]
            if sims:
                best_match = max(sims, key=lambda x: x[1])
                if best_match[1] > 0.6:
                    nome_img = best_match[0].split("_face_")[0]
                    resultados.append((nome_img, best_match[2], face_crop))
    return resultados

def processar_video(video_path, batch_size=15):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imagens_detectadas = {}
    pessoas_encontradas = set()
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

        # Quando atingir o tamanho do lote ou acabar o vídeo
        if len(frames_batch) == batch_size or not cap.isOpened():
            # Processa o lote de frames com YOLO
            yolo_results = yolo.predict(frames_batch, conf=0.5, classes=[0], device="cuda")
            for idx, yolo_result in enumerate(yolo_results):
                frame = frames_batch[idx]
                for box in yolo_result.boxes:
                    if len(pessoas_encontradas) == len(embedding_db):
                        break  # já encontrou todas, pode parar

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_crop = frame[y1:y2, x1:x2]
                    faces = model.get(face_crop)
                    if len(faces) > 0:
                        embedding = faces[0].embedding
                        sims = [
                            (ref["id"], cosine_similarity([ref["embedding"]], [embedding])[0][0], ref["path"])
                            for ref in embedding_db
                            if ref["id"].split("_face_")[0] not in pessoas_encontradas
                        ]
                        if sims:
                            best_match = max(sims, key=lambda x: x[1])
                            if best_match[1] > 0.5:
                                nome_img = best_match[0].split("_face_")[0]
                                pessoas_encontradas.add(nome_img)
                                recorte_nome = f"{nome_img}_frame_{frame_indices[idx]}.jpg"
                                recorte_path = os.path.join(RESULT_FOLDER, recorte_nome)
                                cv2.imwrite(recorte_path, face_crop)
                                imagens_detectadas[nome_img] = {
                                    "referencia": best_match[2],
                                    "recorte": recorte_path
                                }
            frames_batch = []
            frame_indices = []

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
        resultados = processar_frame(frame, embedding_db, pessoas_encontradas)
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
            for nome, dados in imagens_detectadas.items():
                # Copia imagem de referência
                destino_ref = os.path.join(RESULT_FOLDER, os.path.basename(dados["referencia"]))
                if not os.path.exists(destino_ref):
                    os.system(f'cp "{dados["referencia"]}" "{destino_ref}"')
                # Adiciona caminhos para exibição
                imagens_para_exibir.append({
                    "referencia": url_for("static", filename=f"resultados/{os.path.basename(dados["referencia"])}"),
                    "recorte": url_for("static", filename=f"resultados/{os.path.basename(dados["recorte"])}")
                })

            return render_template("resultado.html", imagens=imagens_para_exibir)

    return render_template("index.html")

@app.route("/progresso")
def get_progresso():
    return jsonify(progresso)

@app.route("/processar_frame", methods=["POST"])
def processar_frame_webcam():
    if "frame" not in request.files:
        return jsonify({"mensagem": "Nenhum frame recebido.", "encontrado": False, "pessoas": []})
    file = request.files["frame"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    frame = cv2.imread(filepath)
    pessoas_encontradas = set()
    resultados = processar_frame(frame, embedding_db, pessoas_encontradas)
    pessoas = []
    for nome_img, referencia, face_crop in resultados:
        # Foto recortada do vídeo
        _, buffer = cv2.imencode('.jpg', face_crop)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        # Foto arquivada (referência)
        ref_img = cv2.imread(referencia)
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

if __name__ == "__main__":
    app.run(debug=True)