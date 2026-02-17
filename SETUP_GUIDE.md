# 🚀 Setup Guide - Step by Step

Follow these steps exactly to get your AQI Predictor project running!

## ✅ Step 1: Prerequisites Check

### Install Python
1. Check if Python is installed:
   ```bash
   python --version
   ```
   Should show Python 3.9 or higher.

2. If not installed, download from [python.org](https://www.python.org/downloads/)

### Install Git
1. Check if Git is installed:
   ```bash
   git --version
   ```

2. If not installed, download from [git-scm.com](https://git-scm.com/)

---

## 📁 Step 2: Project Setup

### Create Project Folder
```bash
# Navigate to where you want to create the project
cd Desktop  # or wherever you prefer

# Create and enter project folder
mkdir AQI_Predictor
cd AQI_Predictor
```

### Initialize Git Repository
```bash
git init
```

---

## 🔧 Step 3: Virtual Environment

### Create Virtual Environment
```bash
# Create venv
python -m venv venv
```

### Activate Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## 📦 Step 4: Install Dependencies

### Create requirements.txt
Copy the requirements.txt file provided in the project.

### Install packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will take 5-10 minutes. ☕

---

## 🔑 Step 5: Get API Keys

### 5.1 OpenWeather API Key

1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Click "Sign Up" (top right)
3. Fill in the form:
   - Username: choose any
   - Email: your email
   - Password: choose strong password
   - ✅ Tick "I am 16 years old and over"
   - ✅ Tick "I agree with Privacy Policy..."
   - ❌ Untick newsletter (optional)
4. Click "Create Account"
5. Check your email and verify your account
6. Log in to OpenWeatherMap
7. Click your username (top right) → "My API keys"
8. You'll see a default API key created
9. **Copy this key** - you'll need it soon!

**Important:** It takes about 10 minutes for the API key to activate. Don't worry if it doesn't work immediately!

### 5.2 Hopsworks Account & API Key

1. Go to [Hopsworks](https://app.hopsworks.ai/)
2. Click "Sign Up" or "Get Started"
3. Sign up with:
   - Email and password, OR
   - GitHub account, OR
   - Google account
4. Verify your email if required
5. Once logged in, create a new project:
   - Click "Create New Project"
   - Name: `aqi_predictor`
   - Description: "AQI prediction for Karachi"
   - Click "Create"
6. Generate API key:
   - In your project, click "Settings" (left sidebar)
   - Click "API Keys" tab
   - Click "Generate New Key"
   - Name it: `aqi_pipeline`
   - Click "Create"
   - **Copy the API key** - you can only see it once!

---

## 🔐 Step 6: Configure Environment Variables

### Create .env file

In your project folder, create a file named `.env` (yes, it starts with a dot!):

**On Windows (using notepad):**
```bash
notepad .env
```

**On Mac/Linux:**
```bash
nano .env
```

### Add your API keys

Paste this into the `.env` file and replace the placeholder values:

```env
# OpenWeather API
OPENWEATHER_API_KEY=paste_your_openweather_key_here

# Hopsworks
HOPSWORKS_API_KEY=paste_your_hopsworks_key_here
HOPSWORKS_PROJECT_NAME=aqi_predictor

# Location (Karachi coordinates)
CITY_NAME=Karachi
LATITUDE=24.8607
LONGITUDE=67.0011
```

**Example (with fake keys):**
```env
OPENWEATHER_API_KEY=abc123def456ghi789jkl012mno345pq
HOPSWORKS_API_KEY=xyz789abc123def456ghi789jkl012mno345pqr678stu901
HOPSWORKS_PROJECT_NAME=aqi_predictor
CITY_NAME=Karachi
LATITUDE=24.8607
LONGITUDE=67.0011
```

Save and close the file.

---

## 🧪 Step 7: Test the Setup

### Test 1: Check API Keys

```bash
cd feature_pipeline
python fetch_data.py
```

**Expected output:**
```
Testing Data Fetching Module

==================================================
Fetching current data for Karachi
Location: 24.8607°N, 67.0011°E
==================================================

✓ Weather data fetched successfully for Karachi
✓ Pollution data fetched successfully for Karachi (AQI: 3)

==================================================
✓ All data fetched successfully!
==================================================
```

**If you get errors:**
- `Error fetching pollution data`: Wait 10 minutes, OpenWeather API key is still activating
- `API key not configured`: Check your .env file
- `Module not found`: Make sure you're in venv and installed requirements.txt

### Test 2: Check Hopsworks Connection

```bash
python hopsworks_utils.py
```

**Expected output:**
```
Testing Hopsworks Integration

🔗 Connecting to Hopsworks...
✓ Connected to project: aqi_predictor

✓ Hopsworks integration working!
Project: aqi_predictor
```

---

## 🎉 Step 8: Run First Pipeline

### Collect Initial Data

```bash
# Make sure you're in the feature_pipeline folder
python run_pipeline.py
```

This will:
1. Fetch current weather data ✓
2. Fetch current pollution data ✓
3. Create features ✓
4. Upload to Hopsworks ✓

**Expected output:**
```
============================================================
🚀 FEATURE PIPELINE STARTED
⏰ Time: 2025-01-31 15:30:45
============================================================

Step 1/4: Fetching data from APIs...
✓ Weather data fetched successfully for Karachi
✓ Pollution data fetched successfully for Karachi (AQI: 3)

Step 2/4: Engineering features...
✓ Created 20 features

Step 3/4: Connecting to Hopsworks...
✓ Connected to project: aqi_predictor

Step 4/4: Inserting features into feature store...
⚙ Creating feature group: aqi_features
✓ Successfully inserted 1 records

============================================================
✅ FEATURE PIPELINE COMPLETED SUCCESSFULLY!
============================================================
```

---

## 📊 Step 9: Backfill Historical Data

```bash
python backfill_features.py --days 30
```

This creates training data for the past 30 days.

**Note:** Due to free API limitations, pollution data will be placeholders initially. Real data will accumulate as you run the hourly pipeline!

---

## ✅ Step 10: Verify in Hopsworks

1. Go to [Hopsworks](https://app.hopsworks.ai/)
2. Open your `aqi_predictor` project
3. Click "Feature Store" in left sidebar
4. You should see `aqi_features` feature group
5. Click on it to view your data!

---

## 🎯 Next Steps

Now that setup is complete:

1. ✅ **Data Collection Working** - Pipeline runs successfully
2. ✅ **Feature Store Working** - Data stored in Hopsworks
3. ⏭️ **Next:** Build the training pipeline
4. ⏭️ **Then:** Create the dashboard
5. ⏭️ **Finally:** Automate with GitHub Actions

---

## 🆘 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution:** 
```bash
# Make sure venv is activated
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Problem: "API key not configured"
**Solution:**
- Check that `.env` file exists in project root
- Check that API keys are correct (no quotes, no spaces)
- Make sure `.env` is in the same folder as your Python scripts

### Problem: "Connection refused" or "Network error"
**Solution:**
- Check your internet connection
- Check if firewall is blocking Python
- Try again after a few minutes

### Problem: OpenWeather returns empty data
**Solution:**
- Wait 10-15 minutes after creating API key
- Check you're using the correct API key
- Verify your API key is active on OpenWeather dashboard

### Problem: Hopsworks connection fails
**Solution:**
- Double-check API key (it's very long, copy correctly)
- Make sure project name matches exactly: `aqi_predictor`
- Check Hopsworks is not down: [status.hopsworks.ai](https://status.hopsworks.ai)

---

## 🎊 Setup Complete!

You've successfully:
- ✅ Installed all dependencies
- ✅ Configured API keys
- ✅ Connected to Hopsworks
- ✅ Run your first pipeline
- ✅ Stored data in feature store

**You're ready to build the ML model! 🚀**

---

## 📞 Need Help?

- **Re-read this guide** - Most issues are covered above
- **Check the error message** - It usually tells you what's wrong
- **Google the error** - Someone probably had the same issue
- **Ask your instructor** - They're there to help!

**Common errors are NORMAL - don't give up! 💪**
