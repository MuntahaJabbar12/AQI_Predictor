# ⚡ GitHub Actions - Quick Setup Checklist

## 📋 5-Minute Setup

### ☑️ Step 1: Push to GitHub (2 min)
```bash
cd E:\AQI_Predictor\AQI_Predictor
git init
git add .
git commit -m "Add automation"
git remote add origin https://github.com/YOUR_USERNAME/AQI_Predictor.git
git push -u origin main
```

---

### ☑️ Step 2: Add Secrets (2 min)

Go to: **Repository → Settings → Secrets and variables → Actions**

Add these 3 secrets:

| Name | Value |
|------|-------|
| `OPENWEATHER_API_KEY` | Your OpenWeather key |
| `HOPSWORKS_API_KEY` | Your Hopsworks key |
| `HOPSWORKS_PROJECT_NAME` | AQIPREDICTOR234 |

---

### ☑️ Step 3: Test Run (1 min)

1. Go to **Actions** tab
2. Click **"Hourly Feature Pipeline"**
3. Click **"Run workflow"**
4. Wait 2-3 minutes
5. Check logs for success ✅

---

## 🎯 What Happens Next

### Automatic Schedule:
- **Every hour:** Collects data (24x/day)
- **Every day 2 AM:** Trains models (if data ≥ 50 records)

### Timeline:
- **Day 1:** 24 records
- **Day 3:** 72 records
- **Day 5:** 120 records → **Ready for training!**

---

## 🔍 Quick Checks

### Is it working?
```bash
python check_data.py
```
Should show increasing record count daily.

### View automation:
GitHub → Actions → See all runs

### Fix issues:
Check workflow logs for error messages

---

## 📞 Troubleshooting

| Problem | Solution |
|---------|----------|
| Workflow not running | Enable in Actions tab |
| API key error | Re-check secrets |
| No data added | Check workflow logs |
| Rate limit | Wait 1 hour, retry |

---

## ✅ Success = 
- ✅ Workflow runs hourly
- ✅ No failures
- ✅ Data increases in Hopsworks

---

**That's it! Your automation is live! 🚀**

Check back in 5 days → You'll have 120+ records → Train models!
