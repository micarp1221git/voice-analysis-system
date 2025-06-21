# 🎤 音声診断システム MVP

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

## 🎯 概要
30秒以内の音声ファイルをアップロードして、AI音声診断を受けられるWebアプリケーションです。
歌声、話し声、プレゼンテーション用途に対応した専門的な分析を提供します。

## ✨ 主な機能

### 🎤 音声分析
- **対応形式**: WAV, MP3, M4A, FLAC（30秒制限）
- **6つの評価指標**: 音量、明瞭度、ピッチ安定性など
- **目的別分析**: 歌・話し声・プレゼン対策に最適化
- **星評価システム**: 5段階評価（最大99点）

### 📊 視覚化
- **レーダーチャート**: 6指標の総合評価
- **詳細グラフ**: 波形、スペクトログラム、MFCC、ピッチ変化
- **プログレスバー**: 視覚的なスコア表示
- **レスポンシブデザイン**: モバイル・PC対応

### 🤖 AI診断
- **多様な診断文**: 5段階×複数パターンのバリエーション
- **目的別アドバイス**: 用途に応じたカスタマイズ
- **改善提案**: 具体的で実践的なヒント
- **JPG出力**: SNS投稿用の結果画像生成

### 💼 ビジネス機能
- **入力バリデーション**: 必須項目チェック
- **CTA統合**: 有料コンサルティング誘導
- **ユーザー体験**: 直感的な操作フロー

## 🚀 クイックスタート

### 前提条件
- Python 3.8以上
- pipx または pip

### インストール

#### 方法1: pipx（推奨）
```bash
# pipxのインストール
brew install pipx
pipx ensurepath

# Streamlitのインストール
pipx install streamlit

# 必要なライブラリの追加
pipx inject streamlit librosa numpy matplotlib pandas plotly soundfile scipy pillow

# アプリケーション起動
streamlit run streamlit_app.py
```

#### 方法2: 仮想環境
```bash
# 仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt

# アプリケーション起動
streamlit run streamlit_app.py
```

### アクセス
ブラウザで `http://localhost:8501` を開く

## 🎤 使用方法

1. **情報入力**
   - 名前（必須）
   - 分析目的の選択（話し声の印象/歌/プレゼン対策/その他）

2. **音声アップロード**
   - 対応形式: WAV, MP3, M4A, FLAC
   - 制限: 30秒以内

3. **分析実行**
   - ワンクリックで自動分析
   - リアルタイムプログレス表示

4. **結果確認**
   - 星評価と総合スコア
   - 詳細な項目別評価
   - 具体的な改善アドバイス
   - JPG画像での結果保存

## 🔧 技術スタック

### フロントエンド
- **Streamlit**: Webアプリケーションフレームワーク
- **Plotly**: インタラクティブなグラフ
- **HTML/CSS**: カスタムスタイリング

### バックエンド
- **librosa**: 音声信号処理
- **NumPy**: 数値計算
- **pandas**: データ分析
- **SciPy**: 科学計算
- **Pillow**: 画像生成

### 音声分析機能
- **特徴量抽出**: RMS、ピッチ、MFCC、スペクトラル特徴
- **可視化**: 波形、スペクトログラム、レーダーチャート
- **評価ロジック**: 目的別カスタマイズ

## 🌟 特徴

### 目的別診断
- **歌声**: ピッチ安定性と音域重視
- **話し声**: 明瞭度と聞き取りやすさ重視
- **プレゼン**: 説得力と安定性重視

### ユーザー体験
- **直感的UI**: 3ステップで完了
- **リアルタイム**: 即座にフィードバック
- **視覚的**: グラフとチャートで分かりやすく
- **モバイル対応**: スマートフォンでも利用可能

## 🚀 デプロイ

### Streamlit Cloud（推奨）
1. GitHubリポジトリを作成
2. [Streamlit Cloud](https://share.streamlit.io) にアクセス
3. リポジトリを接続
4. 自動デプロイ開始

## 📈 ロードマップ

### 近日実装予定
- [ ] Whisper音声認識統合
- [ ] GPT API診断文生成
- [ ] ユーザーデータ保存

### 将来的な機能
- [ ] リアルタイム録音
- [ ] 音声比較機能
- [ ] 学習進捗トラッキング
- [ ] 多言語対応

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## 📄 ライセンス

MIT License

## 🙏 謝辞

- **librosa**: 音声処理ライブラリ
- **Streamlit**: 素晴らしいWebアプリフレームワーク

---

⭐ このプロジェクトが役に立ったらスターをお願いします！