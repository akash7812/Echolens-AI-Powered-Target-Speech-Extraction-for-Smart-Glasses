# Echolens — Production & Cloud Deployment Guide
=================================================

This guide explains how to deploy and access the **Echolens Smart-Glasses AI Acoustic Hub** across different environments: Local Network, Instant Public HTTPS Tunnel, Cloud Containers (Hugging Face Spaces / Render), and Windows Desktop.

---

## 1. Local Network Deployment (Wi-Fi / Phone / Smart Glasses)

The Echolens server binds to `0.0.0.0:8000`, making it accessible to any device on the same local network (smartphones, tablets, second laptops, or smart-glasses companion hardware).

### Finding Your Local IP Address
```powershell
ipconfig
# Look for "IPv4 Address" under Wi-Fi (e.g., 10.58.183.74 or 192.168.1.X)
```

### Starting the Server
```powershell
cd C:\Echolens
.\.venv\Scripts\Activate.ps1
python -m echolens server --host 0.0.0.0 --port 8000
```

### URLs for Access
* **On your PC**: `http://localhost:8000` or `http://127.0.0.1:8000`
* **On any device on your Wi-Fi**: `http://10.58.183.74:8000`

---

## 2. Instant 1-Click Desktop Launcher

A 1-click batch launcher has been placed on your **Desktop**:
* **File**: `C:\Users\admin\Desktop\Echolens.bat` (and `C:\Echolens\launch_echolens.bat`)
* **Usage**: Double-click the file. It will automatically:
  1. Activate the virtual environment (`.venv`).
  2. Launch your default browser directly to `http://localhost:8000`.
  3. Start the background Echolens server on `0.0.0.0:8000`.

---

## 3. Instant Public HTTPS Tunnel (Share with Examiners & Test on Mobile)

If you need a public HTTPS link accessible from anywhere outside your local Wi-Fi (e.g. for university evaluators, remote teammates, or phones on cellular data):

### Option A: LocalTunnel (No Installation / No Signup Required)
Run this command in any terminal:
```bash
npx localtunnel --port 8000 --subdomain echolens-demo
```
This generates a live public HTTPS URL:
`https://echolens-demo.loca.lt` (or an auto-generated random URL).

### Option B: Cloudflare Tunnel (Free & Enterprise Grade)
1. Download `cloudflared.exe` from [Cloudflare Releases](https://github.com/cloudflare/cloudflared/releases).
2. Run:
```bash
cloudflared tunnel --url http://127.0.0.1:8000
```
3. Cloudflare gives you a temporary public URL such as `https://xxxx.trycloudflare.com`.

---

## 4. Free Cloud Hosting on Hugging Face Spaces (Permanent URL)

Hugging Face Spaces provides a free 16 GB RAM CPU tier that is ideal for final-year machine learning and audio projects.

### Step-by-Step Setup:
1. Log in to [Hugging Face](https://huggingface.co).
2. Click **New Space** $\rightarrow$ Name it `echolens-smartglasses`.
3. Select **Space SDK**: **Docker** (Blank).
4. Clone the repository to your machine or upload files:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/echolens-smartglasses
   ```
5. Copy the following files into the space repository:
   * `Dockerfile` (already configured in `C:\Echolens\Dockerfile`)
   * `requirements.txt`
   * `pyproject.toml` and `setup.py`
   * `src/` directory
   * `data/` directory
   * Model checkpoint file in `models/` or Hugging Face Model Hub.
6. Commit and push:
   ```bash
   git add .
   git commit -m "Deploy Echolens Smart Glasses Audio Hub"
   git push
   ```
7. Hugging Face will automatically build the container and provide a permanent public link:
   `https://huggingface.co/spaces/<your-username>/echolens-smartglasses`

---

## 5. Cloud Container Deployment (Render / Railway)

Echolens includes a standard [Dockerfile](../Dockerfile) with built-in health checks (`HEALTHCHECK`) and environment variable configuration (`PORT=8000`):

1. Push your repository to GitHub: `git push origin main`.
2. Connect your GitHub repository to [Render](https://render.com) or [Railway](https://railway.app).
3. Select **Web Service** $\rightarrow$ Environment: **Docker**.
4. Set port: `8000`.
5. Deploy!
