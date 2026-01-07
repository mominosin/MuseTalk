# AWS EC2 MuseTalk セットアップガイド

このガイドでは、AWS EC2上でMuseTalkを実行するための完全なセットアップ手順を説明します。

## 目次

1. [EC2インスタンスの選択と起動](#1-ec2インスタンスの選択と起動)
2. [環境構築](#2-環境構築)
3. [MuseTalkのインストール](#3-musetalkのインストール)
4. [モデルのダウンロード](#4-モデルのダウンロード)
5. [推論の実行](#5-推論の実行)
6. [カスタマイズポイント](#6-カスタマイズポイント)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. EC2インスタンスの選択と起動

### 1.1 推奨インスタンスタイプ

| 用途 | インスタンスタイプ | GPU | VRAM | 推定コスト (USD/時) |
|------|-------------------|-----|------|-------------------|
| **開発/テスト** | g4dn.xlarge | T4 x1 | 16GB | ~$0.526 |
| **本番推論** | g4dn.2xlarge | T4 x1 | 16GB | ~$0.752 |
| **高速推論** | g5.xlarge | A10G x1 | 24GB | ~$1.006 |
| **リアルタイム** | p3.2xlarge | V100 x1 | 16GB | ~$3.06 |
| **トレーニング** | p4d.24xlarge | A100 x8 | 320GB | ~$32.77 |

> **推奨**: 推論には `g4dn.xlarge` または `g5.xlarge` を推奨します。

### 1.2 AMIの選択

以下のAMIを推奨します：

```
Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04)
```

または

```
Deep Learning AMI (Amazon Linux 2)
```

### 1.3 EC2インスタンスの起動手順

1. **AWSコンソール** → **EC2** → **インスタンスを起動**

2. **AMI選択**:
   - 「Deep Learning AMI」で検索
   - Ubuntu 20.04 + PyTorch 2.0 を選択

3. **インスタンスタイプ選択**:
   - `g4dn.xlarge` を選択（推奨）

4. **ストレージ設定**:
   - 最低 **100GB** のEBSボリューム（モデル + データ用）
   - gp3タイプを推奨

5. **セキュリティグループ設定**:
   ```
   SSH (22)     : 自分のIP
   HTTP (7860)  : Gradio UI用（必要な場合）
   HTTPS (443)  : 必要な場合
   ```

6. **キーペア**: 既存のものを使用するか新規作成

---

## 2. 環境構築

### 2.1 EC2への接続

```bash
ssh -i "your-key.pem" ubuntu@<EC2-PUBLIC-IP>
```

### 2.2 システムの更新

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.3 CUDA環境の確認

Deep Learning AMIを使用している場合、CUDAは既にインストールされています：

```bash
# CUDA確認
nvidia-smi

# CUDAバージョン確認
nvcc --version
```

### 2.4 FFmpegのインストール

```bash
# FFmpeg 4.4+をインストール
sudo apt install -y ffmpeg

# バージョン確認
ffmpeg -version
```

### 2.5 Conda環境の作成

```bash
# Condaの初期化（Deep Learning AMI使用時）
source ~/anaconda3/etc/profile.d/conda.sh

# MuseTalk専用環境の作成
conda create -n musetalk python=3.10 -y
conda activate musetalk
```

---

## 3. MuseTalkのインストール

### 3.1 リポジトリのクローン

```bash
cd ~
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
```

### 3.2 PyTorchのインストール

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3.3 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3.4 MMLabパッケージのインストール

```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

---

## 4. モデルのダウンロード

### 4.1 自動ダウンロード

```bash
sh download_weights.sh
```

### 4.2 手動ダウンロード（自動が失敗した場合）

```bash
# ディレクトリ作成
mkdir -p models/musetalk models/musetalkV15 models/dwpose \
         models/face-parse-bisent models/sd-vae models/whisper models/syncnet

# Hugging Faceからダウンロード
pip install huggingface_hub

python -c "
from huggingface_hub import hf_hub_download
import os

# MuseTalk v1.0
hf_hub_download(repo_id='TMElyralab/MuseTalk',
                filename='models/musetalk/musetalk.json',
                local_dir='.')
hf_hub_download(repo_id='TMElyralab/MuseTalk',
                filename='models/musetalk/pytorch_model.bin',
                local_dir='.')

# MuseTalk v1.5
hf_hub_download(repo_id='TMElyralab/MuseTalk',
                filename='models/musetalkV15/musetalk.json',
                local_dir='.')
hf_hub_download(repo_id='TMElyralab/MuseTalk',
                filename='models/musetalkV15/unet.pth',
                local_dir='.')
print('Download complete!')
"
```

### 4.3 モデル構造の確認

```bash
ls -la models/
# 以下のディレクトリが存在することを確認:
# musetalk/  musetalkV15/  dwpose/  face-parse-bisent/  sd-vae/  whisper/  syncnet/
```

---

## 5. 推論の実行

### 5.1 Gradio Web UI（推奨）

```bash
# 基本実行
python app.py --use_float16

# 外部アクセスを許可する場合（EC2用）
python app.py --use_float16 --server_name 0.0.0.0 --server_port 7860
```

ブラウザで `http://<EC2-PUBLIC-IP>:7860` にアクセス

### 5.2 コマンドライン推論

#### 通常推論 (v1.5)

```bash
python -m scripts.inference \
    --inference_config configs/inference/test.yaml \
    --result_dir results/test \
    --unet_model_path models/musetalkV15/unet.pth \
    --unet_config models/musetalkV15/musetalk.json \
    --version v15 \
    --use_float16
```

#### 通常推論 (v1.0)

```bash
python -m scripts.inference \
    --inference_config configs/inference/test.yaml \
    --result_dir results/test \
    --unet_model_path models/musetalk/pytorch_model.bin \
    --unet_config models/musetalk/musetalk.json \
    --version v10 \
    --use_float16
```

#### リアルタイム推論

```bash
python -m scripts.realtime_inference \
    --inference_config configs/inference/realtime.yaml \
    --result_dir results/realtime \
    --unet_model_path models/musetalkV15/unet.pth \
    --unet_config models/musetalkV15/musetalk.json \
    --version v15 \
    --fps 25 \
    --use_float16
```

### 5.3 バッチ実行スクリプト

```bash
# v1.5通常推論
sh inference.sh v1.5 normal

# v1.5リアルタイム推論
sh inference.sh v1.5 realtime

# v1.0推論
sh inference.sh v1.0 normal
```

---

## 6. カスタマイズポイント

### 6.1 推論設定 (`configs/inference/test.yaml`)

```yaml
# 複数タスクの定義
task_0:
  video_path: "data/video/sample.mp4"    # 入力動画パス
  audio_path: "data/audio/speech.wav"    # 入力音声パス
  bbox_shift: 0                          # 口の開き具合調整 (-9 ~ +9)

task_1:
  video_path: "data/video/another.mp4"
  audio_path: "data/audio/another.wav"
  bbox_shift: 5                          # 口を大きく開く
```

### 6.2 bbox_shift パラメータ

口の開き具合を調整する重要なパラメータです：

| 値 | 効果 |
|----|------|
| **正の値 (+1 ~ +9)** | 口をより大きく開く |
| **0** | デフォルト |
| **負の値 (-1 ~ -9)** | 口を小さく開く |

> 詳細は `assets/BBOX_SHIFT.md` を参照

### 6.3 リアルタイム設定 (`configs/inference/realtime.yaml`)

```yaml
avator_1:
  preparation: True           # 初回実行時はTrue、2回目以降はFalse
  bbox_shift: 5               # 口の開き具合
  video_path: "data/video/avatar.mp4"
  audio_clips:
    audio_0: "data/audio/clip1.wav"
    audio_1: "data/audio/clip2.wav"
```

### 6.4 パフォーマンス最適化

#### メモリ使用量の削減

```bash
# float16を使用（メモリ半減）
python app.py --use_float16
```

#### 推論速度の向上

```python
# scripts/inference.py の設定例
--batch_size 4              # バッチサイズ調整
--fps 25                    # フレームレート（25fps推奨）
```

### 6.5 入力データの要件

| 項目 | 要件 |
|------|------|
| **動画** | MP4形式、顔が明瞭に映っていること |
| **音声** | WAV 16kHz PCM形式（自動変換あり） |
| **解像度** | 256x256にリサイズされる（内部処理） |
| **FPS** | 25fps推奨 |

### 6.6 モデルバージョン比較

| 特徴 | v1.0 | v1.5 |
|------|------|------|
| **品質** | 良好 | より高品質 |
| **bbox_shift** | 調整可能 | 固定値 |
| **推奨** | テスト用 | **本番用** |
| **同期精度** | 良好 | より精密 |

### 6.7 カスタム入力データの使用

```bash
# 自分の動画・音声でテスト
# 1. 動画をdata/video/に配置
cp your_video.mp4 data/video/

# 2. 音声をdata/audio/に配置
cp your_audio.wav data/audio/

# 3. 設定ファイルを編集
vim configs/inference/test.yaml
```

### 6.8 出力のカスタマイズ

```bash
# 結果の出力先変更
python -m scripts.inference \
    --result_dir /path/to/output \
    ...
```

### 6.9 Gradio UIのカスタマイズ

`app.py` の主要パラメータ：

```python
--server_name 0.0.0.0    # 外部アクセス許可
--server_port 7860       # ポート番号
--share                  # Gradio共有リンク生成
--use_float16            # 省メモリモード
```

---

## 7. トラブルシューティング

### 7.1 CUDA関連エラー

```bash
# CUDA確認
nvidia-smi

# PyTorchでCUDA確認
python -c "import torch; print(torch.cuda.is_available())"
```

### 7.2 メモリ不足 (OOM)

```bash
# float16を使用
python app.py --use_float16

# バッチサイズを小さくする
# configs内のbatch_sizeを減らす
```

### 7.3 FFmpegエラー

```bash
# FFmpegパス確認
which ffmpeg

# 環境変数設定
export FFMPEG_PATH=$(which ffmpeg)
```

### 7.4 モデルダウンロードエラー

```bash
# Hugging Face認証（大容量モデル用）
pip install huggingface_hub
huggingface-cli login
```

### 7.5 権限エラー

```bash
# スクリプトに実行権限を付与
chmod +x download_weights.sh
chmod +x inference.sh
```

---

## クイックスタートまとめ

```bash
# 1. EC2接続
ssh -i key.pem ubuntu@<IP>

# 2. 環境準備
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n musetalk python=3.10 -y && conda activate musetalk

# 3. インストール
git clone https://github.com/TMElyralab/MuseTalk.git && cd MuseTalk
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -U openmim && mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"

# 4. モデルダウンロード
sh download_weights.sh

# 5. 実行
python app.py --use_float16 --server_name 0.0.0.0
```

---

## 参考リンク

- [MuseTalk GitHub](https://github.com/TMElyralab/MuseTalk)
- [技術論文](https://arxiv.org/abs/2410.10122)
- [Hugging Face Models](https://huggingface.co/TMElyralab/MuseTalk)
- [AWS EC2 GPU インスタンス](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
