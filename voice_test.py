import librosa
import numpy as np
import os

print("🎤 音声診断システム - ライブテスト")
print("=" * 50)

# 音声ファイル確認
test_file = "AI音声シャドーイング用.wav"
if not os.path.exists(test_file):
    print("❌ テスト用音声ファイルが見つかりません")
    exit(1)

print(f"📁 テストファイル: {test_file}")

# 音声分析実行
try:
    y, sr = librosa.load(test_file)
    duration = len(y) / sr
    
    print(f"✅ 音声読み込み成功")
    print(f"   長さ: {duration:.2f}秒")
    print(f"   サンプリングレート: {sr}Hz")
    
    # 特徴量抽出
    rms = librosa.feature.rms(y=y)[0]
    avg_volume = np.mean(rms)
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = [pitches[magnitudes[:, t].argmax(), t] 
                   for t in range(pitches.shape[1]) 
                   if pitches[magnitudes[:, t].argmax(), t] > 0]
    
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    print(f"\n📊 分析結果:")
    print(f"   音量指標: {avg_volume:.6f}")
    print(f"   ピッチ点数: {len(pitch_values)}")
    if pitch_values:
        print(f"   平均ピッチ: {np.mean(pitch_values):.1f}Hz")
        print(f"   ピッチ範囲: {np.min(pitch_values):.1f}-{np.max(pitch_values):.1f}Hz")
    print(f"   音の明るさ: {np.mean(spectral_centroids):.1f}Hz")
    print(f"   明瞭度: {np.mean(zcr):.6f}")
    
    # 簡易診断
    print(f"\n🤖 簡易診断:")
    if avg_volume > 0.05:
        print("✅ 十分な音量で話せています")
    else:
        print("⚠️  もう少し大きな声で話すと良いでしょう")
    
    if len(pitch_values) > 50:
        pitch_stability = 1.0 / (1.0 + np.std(pitch_values)/np.mean(pitch_values))
        if pitch_stability > 0.7:
            print("✅ ピッチが安定しており、聞き取りやすい声です")
        else:
            print("⚠️  ピッチを安定させると印象が向上します")
    
    if np.mean(zcr) > 0.1:
        print("✅ 明瞭な発音ができています")
    else:
        print("⚠️  発音の明瞭度を向上させると良いでしょう")
    
    print(f"\n🎯 30秒制限チェック: {'✅ OK' if duration <= 30 else '❌ 超過'}")
    
    print(f"\n🚀 コア機能テスト完了\!")
    print("💡 次のステップ: Streamlitアプリの実行")
    
except Exception as e:
    print(f"❌ エラー: {e}")

EOF < /dev/null