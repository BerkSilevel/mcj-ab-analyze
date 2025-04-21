from model import DecayRetentionModel
import joblib

# Modeli oluştur
model = DecayRetentionModel()

# Modeli kaydet (fit gerekmez çünkü sabit bir formül)
joblib.dump(model, "model_d4_d15.pkl")

print("✅ Model başarıyla kaydedildi: model_d4_d15.pkl")
