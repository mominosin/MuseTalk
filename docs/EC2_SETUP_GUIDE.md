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
Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
```

または

```
Ubuntu Server 22.04 LTS
```

> **注意**: Deep Learning AMIを使用する場合でも、CUDAの追加インストールが必要な場合があります。

### 1.3 EC2インスタンスの起動手順

1. **AWSコンソール** → **EC2** → **インスタンスを起動**

2. **AMI選択**:
   - 「Ubuntu 22.04」で検索
   - GPU対応AMIを選択

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

### 2.3 CUDA環境のセットアップ

#### 2.3.1 GPU確認

```bash
nvidia-smi
```

#### 2.3.2 CUDA Toolkit 11.8 のインストール

Ubuntu 22.04では`libtinfo5`の依存関係問題が発生する場合があります。以下の手順で解決してください：

```bash
# libtinfo5 をインストール（依存関係問題の解決）
wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb

# NVIDIAリポジトリの追加
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# CUDA Toolkit インストール
sudo apt install -y cuda-toolkit-11-8
```

#### 2.3.3 環境変数の設定

```bash
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 確認
nvcc --version
```

### 2.4 FFmpegのインストール

```bash
sudo apt install -y ffmpeg

# バージョン確認
ffmpeg -version
```

### 2.5 Minicondaのインストール

```bash
# Minicondaをダウンロード・インストール
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init
source ~/.bashrc
```

### 2.6 Conda環境の作成

#### 2.6.1 利用規約への同意（必要な場合）

```bash
# Anacondaの利用規約に同意が必要な場合
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

または、conda-forgeを使用して利用規約を回避：

```bash
# conda-forgeのみを使用する場合
conda create -n musetalk python=3.10 -c conda-forge --override-channels -y
```

#### 2.6.2 環境作成（通常）

```bash
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

### 3.3 基本パッケージのインストール

```bash
# pip/setuptoolsを最新化
pip install --upgrade pip setuptools wheel

# 依存パッケージをインストール
pip install -r requirements.txt
```

### 3.4 MMLabパッケージのインストール

```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
```

#### 3.4.1 mmpose のインストール（chumpy問題の回避）

mmpose インストール時に`chumpy`のビルドエラーが発生する場合があります：

```bash
# 方法1: --no-deps でインストール（推奨）
pip install mmpose==1.1.0 --no-deps

# 必要な依存関係を個別インストール
pip install xtcocotools munkres json_tricks

# 方法2: chumpy を --no-build-isolation でインストール
# pip install chumpy --no-build-isolation
# mim install "mmpose==1.1.0"
```

> **注意**: `chumpy`はSMPLボディモデル用で、MuseTalkの顔リップシンクには不要です。

---

## 4. モデルのダウンロード

### 4.1 huggingface-cli のインストール

```bash
pip install -U "huggingface_hub[cli]"
```

### 4.2 ディレクトリ作成

```bash
mkdir -p models/musetalk models/musetalkV15 models/syncnet \
         models/dwpose models/face-parse-bisent models/sd-vae models/whisper
```

### 4.3 モデルのダウンロード

```bash
# MuseTalk V1.0
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir models \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# MuseTalk V1.5
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir models \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# SD VAE
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir models/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Whisper
huggingface-cli download openai/whisper-tiny \
  --local-dir models/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# DWPose
huggingface-cli download yzd-v/DWPose \
  --local-dir models/dwpose \
  --include "dw-ll_ucoco_384.pth"

# SyncNet
huggingface-cli download ByteDance/LatentSync \
  --local-dir models/syncnet \
  --include "latentsync_syncnet.pt"
```

### 4.4 Face Parse モデルのダウンロード

```bash
pip install gdown

# Face Parse BiSeNet
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth

# ResNet18
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o models/face-parse-bisent/resnet18-5c106cde.pth
```

### 4.5 自動ダウンロードスクリプト（代替）

```bash
# スクリプトに実行権限を付与
chmod +x download_weights.sh

# 実行（中国ミラーを使用）
sh download_weights.sh
```

> **注意**: `download_weights.sh`は中国のHugging Faceミラー（`hf-mirror.com`）を使用します。日本からアクセスする場合は、スクリプト内の`export HF_ENDPOINT=https://hf-mirror.com`をコメントアウトまたは削除してください。

### 4.6 モデル構造の確認

```bash
ls -la models/
# 以下のディレクトリとファイルが存在することを確認:
# musetalk/      - musetalk.json, pytorch_model.bin
# musetalkV15/   - musetalk.json, unet.pth
# dwpose/        - dw-ll_ucoco_384.pth
# face-parse-bisent/ - 79999_iter.pth, resnet18-5c106cde.pth
# sd-vae/        - config.json, diffusion_pytorch_model.bin
# whisper/       - config.json, pytorch_model.bin, preprocessor_config.json
# syncnet/       - latentsync_syncnet.pt
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
chmod +x inference.sh

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

```bash
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

```bash
--server_name 0.0.0.0    # 外部アクセス許可
--server_port 7860       # ポート番号
--share                  # Gradio共有リンク生成
--use_float16            # 省メモリモード
```

---

## 7. トラブルシューティング

### 7.1 CUDA Toolkit インストール時の libtinfo5 エラー

**エラー内容**:
```
nsight-systems-2022.4.2 : Depends: libtinfo5 but it is not installable
```

**解決策**:
```bash
# libtinfo5 を手動インストール
wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb

# その後、CUDA Toolkitを再インストール
sudo apt install -y cuda-toolkit-11-8
```

**代替策（Conda経由）**:
```bash
conda activate musetalk
conda install -c nvidia cuda-toolkit=11.8 -y
```

### 7.2 Conda 利用規約エラー

**エラー内容**:
```
CondaToSNonInteractiveError: Terms of Service have not been accepted
```

**解決策**:
```bash
# 利用規約に同意
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

**代替策（conda-forge使用）**:
```bash
conda create -n musetalk python=3.10 -c conda-forge --override-channels -y
```

### 7.3 chumpy ビルドエラー（mmpose インストール時）

**エラー内容**:
```
ModuleNotFoundError: No module named 'pip'
ERROR: Failed to build 'chumpy' when getting requirements to build wheel
```

**解決策**:
```bash
# mmpose を依存関係なしでインストール
pip install mmpose==1.1.0 --no-deps

# 必要な依存関係のみ個別インストール
pip install xtcocotools munkres json_tricks
```

> `chumpy`はSMPLボディモデル用で、MuseTalkには不要です。

### 7.4 huggingface-cli: command not found

**解決策**:
```bash
pip install -U "huggingface_hub[cli]"
```

### 7.5 Hugging Face 404 エラー

**エラー内容**:
```
Entry Not Found for url: https://huggingface.co/...
```

**解決策**:
`huggingface-cli download`コマンドを使用し、正しいファイルパスを指定：

```bash
# 正しいコマンド形式
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir models \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"
```

### 7.6 CUDA関連エラー

```bash
# CUDA確認
nvidia-smi

# PyTorchでCUDA確認
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

### 7.7 メモリ不足 (OOM)

```bash
# float16を使用
python app.py --use_float16

# バッチサイズを小さくする
# configs内のbatch_sizeを減らす
```

### 7.8 FFmpegエラー

```bash
# FFmpegパス確認
which ffmpeg

# 環境変数設定
export FFMPEG_PATH=$(which ffmpeg)
```

### 7.9 権限エラー

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

# 2. システム準備
sudo apt update && sudo apt upgrade -y
sudo apt install -y ffmpeg

# 3. libtinfo5 インストール（CUDA依存関係）
wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb

# 4. Conda環境準備
conda create -n musetalk python=3.10 -c conda-forge --override-channels -y
conda activate musetalk

# 5. リポジトリクローン
git clone https://github.com/TMElyralab/MuseTalk.git && cd MuseTalk

# 6. パッケージインストール
pip install --upgrade pip setuptools wheel
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -U openmim && mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0"
pip install mmpose==1.1.0 --no-deps && pip install xtcocotools munkres json_tricks

# 7. モデルダウンロード
pip install -U "huggingface_hub[cli]" gdown
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

huggingface-cli download TMElyralab/MuseTalk --local-dir models --include "musetalk/*" "musetalkV15/*"
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae
huggingface-cli download openai/whisper-tiny --local-dir models/whisper
huggingface-cli download yzd-v/DWPose --local-dir models/dwpose --include "dw-ll_ucoco_384.pth"
huggingface-cli download ByteDance/LatentSync --local-dir models/syncnet --include "latentsync_syncnet.pt"
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth

# 8. 実行
python app.py --use_float16 --server_name 0.0.0.0
```

---

## 参考リンク

- [MuseTalk GitHub](https://github.com/TMElyralab/MuseTalk)
- [技術論文](https://arxiv.org/abs/2410.10122)
- [Hugging Face Models](https://huggingface.co/TMElyralab/MuseTalk)
- [AWS EC2 GPU インスタンス](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
