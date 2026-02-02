# 🤖 Phase 3: CI/CD Automation - Complete Setup Guide

This guide will help you set up **automatic hourly data collection** and **daily model training** using GitHub Actions!

---

## 🎯 What You'll Achieve

After setup:
- ✅ **Automatic data collection** every hour (24 times/day)
- ✅ **Automatic model training** every day at 2 AM
- ✅ **No manual work** - everything runs in the cloud
- ✅ **Real data accumulation** for 3-5 days
- ✅ **Ready for Phase 2** (model training) after data collection

---

## 📋 Prerequisites

Before starting, make sure you have:
- ✅ GitHub account
- ✅ Your project code ready
- ✅ OpenWeather API key
- ✅ Hopsworks API key
- ✅ Feature pipeline tested locally

---

## 🚀 Step-by-Step Setup

### Step 1: Push Your Code to GitHub

#### 1.1 Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click **"+"** (top right) → **"New repository"**
3. Repository name: `AQI_Predictor`
4. Description: "Air Quality Index prediction for Karachi"
5. **Keep it Private** (recommended)
6. **Don't** initialize with README (we already have one)
7. Click **"Create repository"**

#### 1.2 Push Your Code

Open terminal in your project folder:

```bash
cd E:\AQI_Predictor\AQI_Predictor

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - AQI Predictor with automation"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/AQI_Predictor.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Enter your GitHub credentials when prompted.**

---

### Step 2: Add GitHub Secrets

GitHub Secrets keep your API keys safe and private.

#### 2.1 Navigate to Secrets

1. Go to your repository on GitHub
2. Click **"Settings"** (top right)
3. In left sidebar, click **"Secrets and variables"** → **"Actions"**
4. Click **"New repository secret"**

#### 2.2 Add These 3 Secrets

**Secret 1: OPENWEATHER_API_KEY**
- Name: `OPENWEATHER_API_KEY`
- Value: (paste your OpenWeather API key)
- Click **"Add secret"**

**Secret 2: HOPSWORKS_API_KEY**
- Name: `HOPSWORKS_API_KEY`
- Value: (paste your Hopsworks API key)
- Click **"Add secret"**

**Secret 3: HOPSWORKS_PROJECT_NAME**
- Name: `HOPSWORKS_PROJECT_NAME`
- Value: `AQIPREDICTOR234` (or your project name)
- Click **"Add secret"**

#### 2.3 Verify Secrets

You should see 3 secrets listed:
- ✅ OPENWEATHER_API_KEY
- ✅ HOPSWORKS_API_KEY
- ✅ HOPSWORKS_PROJECT_NAME

**⚠️ IMPORTANT:** You can't view secret values after creation - they're hidden for security!

---

### Step 3: Enable GitHub Actions

#### 3.1 Check Workflows

1. In your repository, click **"Actions"** tab (top menu)
2. You should see 2 workflows:
   - 🟢 **Hourly Feature Pipeline**
   - 🟢 **Daily Model Training**

#### 3.2 Enable Workflows

If you see "Workflows disabled":
1. Click **"I understand my workflows, go ahead and enable them"**
2. Workflows are now active!

---

### Step 4: Test Your Automation

Don't wait for the schedule! Test manually first.

#### 4.1 Test Feature Pipeline

1. Go to **Actions** tab
2. Click **"Hourly Feature Pipeline"** (left sidebar)
3. Click **"Run workflow"** button (right side)
4. Select branch: `main`
5. Click green **"Run workflow"**

**What happens:**
- GitHub spins up a virtual machine
- Installs Python and dependencies
- Runs your feature pipeline
- Collects data and uploads to Hopsworks

#### 4.2 Monitor Progress

1. You'll see a workflow run appear
2. Click on it to see details
3. Click **"collect-data"** to see logs
4. Expand steps to see output

**Expected output:**
```
✓ Weather data fetched successfully for Karachi
✓ Pollution data fetched successfully for Karachi
✓ Connected to project: AQIPREDICTOR234
✓ Successfully inserted 1 records
```

#### 4.3 Verify in Hopsworks

1. Go to Hopsworks
2. Check Feature Store → aqi_features
3. You should see a new record added!

---

### Step 5: Understanding the Schedule

#### Hourly Feature Pipeline
```yaml
cron: '0 * * * *'
```
Runs at **minute 0 of every hour**:
- 00:00, 01:00, 02:00, ..., 23:00

That's **24 times per day**!

#### Daily Training Pipeline
```yaml
cron: '0 2 * * *'
```
Runs at **2:00 AM UTC** every day.

**Note:** Training only runs if you have **50+ records** with real AQI data.

---

### Step 6: Monitor Your Automation

#### 6.1 Check Workflow Runs

Go to **Actions** tab anytime to see:
- ✅ Recent runs
- ⏰ When they ran
- ✓ Success/failure status
- 📊 Logs

#### 6.2 Email Notifications

GitHub sends email if workflows fail:
- Check your email
- Fix any issues
- Re-run manually

#### 6.3 Check Data Growth

Every few days, run locally:
```bash
python check_data.py
```

You should see records increasing:
- Day 1: ~24 records
- Day 2: ~48 records
- Day 3: ~72 records
- Day 5: ~120 records ✅ (enough for training!)

---

## 📊 Data Collection Timeline

| Day | Records | Status |
|-----|---------|--------|
| Day 1 | ~24 | 🟡 Collecting... |
| Day 2 | ~48 | 🟡 Collecting... |
| Day 3 | ~72 | 🟢 Can start basic training |
| Day 5 | ~120 | 🟢 Good training data |
| Day 7 | ~168 | 🟢 Excellent data |

---

## 🎯 When to Move to Phase 2

**Minimum requirements for model training:**
- ✅ At least **50 records** with real AQI data
- ✅ At least **3 days** of continuous data
- ✅ No major gaps in data

**Recommended:**
- ✅ **100+ records** (5 days)
- ✅ **Covers different hours** (rush hour, night, etc.)
- ✅ **Different days** (weekday, weekend)

---

## 🛠️ Troubleshooting

### Issue: Workflow doesn't run

**Check:**
1. Go to Actions → Hourly Feature Pipeline
2. Is it enabled? (should show green dot)
3. Click "Enable workflow" if disabled

### Issue: Workflow fails with "API key error"

**Fix:**
1. Go to Settings → Secrets
2. Check all 3 secrets are added
3. Re-create the secret if needed
4. Names must match EXACTLY (case-sensitive)

### Issue: "Rate limit exceeded"

**Reason:** OpenWeather free tier = 1,000 calls/day
**Solution:** 
- Hourly calls = 24/day (well under limit)
- Wait an hour and try again
- Check you're not running manually too much

### Issue: Workflow succeeds but no data in Hopsworks

**Debug:**
1. Check workflow logs in GitHub Actions
2. Look for error messages
3. Verify Hopsworks API key is correct
4. Test locally: `python run_pipeline.py`

### Issue: Training workflow always skips

**Reason:** Not enough real data yet (< 50 records)
**Solution:** Wait 3-5 more days for data to accumulate

---

## 📈 Optimization Tips

### 1. Reduce API Calls

If you want to save API calls:
```yaml
# Run every 2 hours instead
cron: '0 */2 * * *'

# Or run only during important hours (6 AM - 10 PM)
cron: '0 6-22 * * *'
```

### 2. Add Notifications

Get notified on Slack/Discord when pipelines run:
- Use GitHub Actions notification integrations
- Add webhook actions to workflows

### 3. Run Multiple Cities

Modify workflows to collect data for multiple cities:
- Duplicate the collection step
- Change CITY_NAME, LATITUDE, LONGITUDE

---

## 🎓 Understanding GitHub Actions

### What are GitHub Actions?

Free CI/CD platform by GitHub that:
- Runs code in the cloud
- Triggers on schedule or events
- 2,000 free minutes/month
- Perfect for ML pipelines!

### How much does it cost?

**FREE** for public repos!
**FREE** for private repos (2,000 minutes/month)

Each workflow run takes ~5 minutes:
- 24 hourly runs = 120 min/day
- 1 daily training = 10 min/day
- **Total: 130 min/day = ~4,000 min/month**

**⚠️ Slightly over free tier!**

**Solutions:**
1. Make repo public (unlimited free minutes)
2. Run every 2 hours (60 min/day)
3. Pause during weekends

---

## 📊 Monitoring Dashboard

Create a simple monitoring script:

```bash
python check_data.py
```

Shows:
- Total records
- Records today
- AQI distribution
- Data quality

Run this **every 2-3 days** to monitor progress.

---

## ✅ Success Checklist

Your automation is working when:

- ✅ GitHub Actions runs every hour
- ✅ No workflow failures
- ✅ New data appears in Hopsworks hourly
- ✅ Email notifications work (if set up)
- ✅ Data count increases daily

---

## 🎯 Next Steps

### Now (Day 1):
- ✅ Set up GitHub Actions
- ✅ Test manual run
- ✅ Verify data collection works

### Days 1-5:
- ⏰ Let automation run
- 📊 Check progress every 2 days
- 🛠️ Fix any issues

### Day 5-7:
- ✅ You should have 100+ records
- 🎯 **Move to Phase 2** (Model Training)
- 🚀 Train models with real data!

### After Training:
- 🎨 **Phase 4:** Build Dashboard
- 🌐 **Phase 5:** Deploy to production

---

## 💡 Pro Tips

1. **Don't test too frequently** - Respect API rate limits
2. **Check logs regularly** - Catch issues early
3. **Document everything** - For your final report
4. **Save workflow runs** - Show your instructor
5. **Monitor Hopsworks quota** - Free tier has limits

---

## 🆘 Need Help?

### Common Commands

**Check data locally:**
```bash
python check_data.py
```

**Test pipeline locally:**
```bash
cd feature_pipeline
python run_pipeline.py
```

**View logs:**
- GitHub → Actions → Click workflow run

**Update code:**
```bash
git add .
git commit -m "Update pipeline"
git push
```

---

## 🎉 Congratulations!

You've set up a **production-grade ML pipeline** with:
- ✅ Automated data collection
- ✅ Cloud infrastructure
- ✅ Version control
- ✅ CI/CD automation
- ✅ Monitoring

**This is exactly how real ML systems work!** 🚀

---

## 📝 What to Document

For your final report, capture:
- Screenshots of GitHub Actions running
- Workflow YAML files explanation
- Data growth over time
- Any issues faced and solutions
- Why automation matters in ML

---

**Now sit back and let the automation do its magic!** ✨

In 5-7 days, you'll have enough data to train amazing models! 💪
