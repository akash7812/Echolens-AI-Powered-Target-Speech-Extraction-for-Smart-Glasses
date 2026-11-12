# Echolens — 1-Click Render Deployment Guide
=============================================

This guide provides step-by-step instructions for deploying the **Echolens Smart-Glasses AI Acoustic Hub** to **Render** ([render.com](https://render.com)).

---

## ⚡ Why Render Instead of Vercel?

| Feature | Render (Recommended) | Vercel |
| :--- | :--- | :--- |
| **PyTorch & Audio AI** | ✅ **Full Support** (Native Docker containers) | ❌ **Unsupported** (Fails 250 MB size limit; PyTorch is 2.2 GB) |
| **Execution Timeout** | ✅ **Continuous background service** | ❌ **10s - 15s timeout** (Causes 504 Gateway errors during inference) |
| **Libsndfile / C Audio** | ✅ **Full Linux support** | ❌ **Restricted serverless runtime** |
| **Pricing** | ✅ **Free Tier** | ✅ Free Tier |

> [!NOTE]
> **Vercel** is designed for static frontends (Next.js, React). Full-stack Python deep learning applications with PyTorch models require persistent containers like **Render**.

---

## 🚀 3-Minute Deployment Steps on Render

### Step 1: Push Your Code to GitHub

Open PowerShell in `C:\Echolens` and run:

```powershell
cd C:\Echolens
git add .
git commit -m "Deploy: Add Render blueprint, Dockerfile, and clean UI"
git push origin main
```

*(Your repository is connected to: `https://github.com/akash7812/Echolens-AI-Powered-Target-Speech-Extraction-for-Smart-Glasses`)*

---

### Step 2: Create a Free Service on Render

1. Open **[dashboard.render.com](https://dashboard.render.com)** and sign in (using GitHub).
2. Click the blue **New +** button in the top right $\rightarrow$ Select **Web Service**.
3. Choose **Build and deploy from a Git repository**.
4. Select your repository: `akash7812/Echolens-AI-Powered-Target-Speech-Extraction-for-Smart-Glasses`.

---

### Step 3: Configure Settings (Render Auto-Detects Most Fields)

Render will automatically detect your `Dockerfile`:

| Setting | Value |
| :--- | :--- |
| **Name** | `echolens` (or your choice) |
| **Region** | `Oregon (US West)` or `Frankfurt (EU)` |
| **Branch** | `main` |
| **Runtime** | `Docker` |
| **Instance Type** | **Free** ($0 / month) |

Click **Create Web Service**.

---

### Step 4: Live URL & Continuous Deployment

* Render will start building the Docker container (installing Python 3.12, PyTorch CPU, and libsndfile).
* Once the build completes (approx. 2-3 minutes), your app is live!
* You will get a permanent public HTTPS URL:
  **`https://echolens-xxxx.onrender.com`**
* Any future `git push origin main` will automatically trigger a redeployment!
