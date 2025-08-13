# Hybrid Architecture Setup Guide

Deploy the AI Research Agent with a **free frontend** on Streamlit Cloud and a **paid backend** on cloud infrastructure.

## 🏗️ Architecture Overview

```
Frontend (Free)           Backend (Paid)
┌─────────────────┐       ┌──────────────────┐
│  Streamlit App  │────►  │   FastAPI Server │
│  (frontend_     │       │   (backend_api   │
│   hybrid.py)    │       │    .py)         │
│                 │       │                  │
│  Streamlit      │       │  DigitalOcean/   │
│  Cloud (Free)   │       │  AWS/GCP/Heroku  │
└─────────────────┘       └──────────────────┘
         │                          │
         └──────── HTTPS API ────────┘
```

**Benefits:**
- ✅ **Free frontend hosting** on Streamlit Cloud
- ✅ **Full system capabilities** on paid backend
- ✅ **Better resource management** and scalability
- ✅ **Separate scaling** of frontend and backend

---

## 🚀 Quick Start

### Step 1: Deploy Backend API

#### Option A: DigitalOcean App Platform (Recommended)
```bash
# 1. Create DigitalOcean account
# 2. Create new App from GitHub repo
# 3. Select backend_api.py as entry point
# 4. Add environment variables:
OPENAI_API_KEY=your-openai-key
SEMANTIC_SCHOLAR_API_KEY=your-optional-key
NEWS_API_KEY=your-optional-key

# 5. Deploy (takes 5-10 minutes)
```

#### Option B: Heroku
```bash
# 1. Install Heroku CLI
# 2. Create new app
heroku create your-backend-app-name

# 3. Set environment variables
heroku config:set OPENAI_API_KEY="your-key"

# 4. Create Procfile
echo "web: python backend_api.py" > Procfile

# 5. Deploy
git add .
git commit -m "Deploy backend API"
git push heroku main
```

#### Option C: Docker + Cloud Run
```bash
# 1. Build Docker image
docker build -f Dockerfile.backend -t research-backend .

# 2. Push to container registry
docker tag research-backend gcr.io/your-project/research-backend
docker push gcr.io/your-project/research-backend

# 3. Deploy to Cloud Run
gcloud run deploy research-backend \
  --image gcr.io/your-project/research-backend \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY="your-key"
```

### Step 2: Deploy Frontend

1. **Fork this repository** to your GitHub account

2. **Create Streamlit Cloud account** at https://share.streamlit.io/

3. **Connect your repository** and select `frontend_hybrid.py`

4. **Add secrets** in Streamlit Cloud settings:
   ```toml
   BACKEND_API_URL = "https://your-backend-api.herokuapp.com"
   ```

5. **Deploy** - Your frontend is now live and free!

---

## 🔧 Detailed Setup Instructions

### Backend Deployment

#### Environment Variables Required
```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional (for full system)
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-key
NEWS_API_KEY=your-news-api-key

# Optional (system config)
PORT=8000
ENVIRONMENT=production
```

#### Backend Dependencies
The backend uses `requirements-backend.txt` which includes:
- FastAPI for API endpoints
- Full research system dependencies
- ChromaDB for vector storage
- All external API integrations

#### Backend Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /status` - System status
- `POST /research` - Conduct research
- `GET /sessions/{id}` - Get session results
- `GET /systems/available` - Available systems

### Frontend Configuration

#### Frontend Dependencies  
The frontend uses minimal dependencies:
```txt
streamlit>=1.28.0
requests>=2.31.0
```

#### Streamlit Secrets Configuration
```toml
# Required: Backend API URL
BACKEND_API_URL = "https://your-backend-api.com"

# Optional: Additional configuration
[frontend_config]
timeout_seconds = 300
max_sessions_history = 10
enable_debug = false
```

---

## 💰 Cost Analysis

### Monthly Costs

| Component | Service | Cost | Notes |
|-----------|---------|------|-------|
| **Frontend** | Streamlit Cloud | $0 | Free tier |
| **Backend** | DigitalOcean Basic | $12 | 2GB RAM, 1 vCPU |
| **Backend** | Heroku Basic | $7 | 512MB RAM (limited) |
| **Backend** | AWS EC2 t3.small | $15 | 2GB RAM, 2 vCPUs |
| **API Usage** | OpenAI GPT-4o-mini | $5-20 | Based on usage |
| **Total** | - | **$12-35/month** | Plus API costs |

### Cost Optimization Tips
1. **Use GPT-4o-mini** for all requests (cheapest model)
2. **Implement request caching** to reduce API calls
3. **Set token limits** to control costs
4. **Monitor usage** with built-in tracking
5. **Use lightweight system** when possible

---

## 🛠️ Platform-Specific Instructions

### DigitalOcean App Platform
```yaml
# .do/app.yaml
name: research-backend
services:
- name: api
  source_dir: /
  github:
    repo: your-username/your-repo
    branch: main
  run_command: python backend_api.py
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: OPENAI_API_KEY
    scope: RUN_TIME
    type: SECRET
    value: your-api-key
```

### AWS Elastic Beanstalk
```python
# application.py (for Elastic Beanstalk)
from backend_api import app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
```

### Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-f', 'Dockerfile.backend', '-t', 
         'gcr.io/$PROJECT_ID/research-backend', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/research-backend']
- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - 'run'
  - 'deploy'
  - 'research-backend'
  - '--image'
  - 'gcr.io/$PROJECT_ID/research-backend'
  - '--region'
  - 'us-central1'
  - '--platform'
  - 'managed'
```

---

## 🔒 Security Configuration

### API Security
```python
# In backend_api.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.streamlit.app"],  # Specific domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Environment Security
- ✅ Use environment variables for API keys
- ✅ Enable HTTPS on backend
- ✅ Configure CORS properly
- ✅ Implement rate limiting
- ✅ Monitor API usage

### Production Checklist
- [ ] API keys in environment variables
- [ ] HTTPS enabled on backend
- [ ] CORS configured for frontend domain
- [ ] Rate limiting implemented
- [ ] Error handling and logging
- [ ] Health checks configured
- [ ] Monitoring and alerts set up

---

## 📊 Monitoring and Maintenance

### Backend Monitoring
```python
# Add to backend_api.py
import logging
from datetime import datetime

# Log all research requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"{request.method} {request.url} - {response.status_code} - {duration:.2f}s")
    return response
```

### Cost Monitoring
```python
# Track API costs
class CostTracker:
    def __init__(self):
        self.daily_cost = 0.0
        self.monthly_limit = 50.0  # $50 limit
    
    def track_request(self, tokens_used: int):
        cost = (tokens_used / 1000) * 0.0015  # GPT-4o-mini pricing
        self.daily_cost += cost
        
        if self.daily_cost > self.monthly_limit / 30:
            # Alert or throttle requests
            pass
```

### Health Monitoring
- **Backend uptime monitoring**
- **API response time tracking** 
- **Error rate monitoring**
- **Resource usage alerts**
- **Cost threshold alerts**

---

## 🚨 Troubleshooting

### Common Issues

#### Backend Not Responding
```bash
# Check backend health
curl https://your-backend.com/health

# Check logs
heroku logs --tail  # For Heroku
```

#### CORS Errors
```python
# Fix CORS in backend_api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-streamlit-app.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### High API Costs
1. **Check token usage** in backend logs
2. **Reduce max_tokens** limits
3. **Implement caching** for common queries
4. **Use shorter context** prompts

#### Memory Issues
1. **Upgrade backend instance** size
2. **Implement result cleanup**
3. **Limit concurrent sessions**
4. **Use lightweight system** mode

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🔄 Updates and Scaling

### Updating the System
1. **Backend updates** - Deploy new version to cloud service
2. **Frontend updates** - Streamlit Cloud auto-deploys from GitHub
3. **Dependencies** - Update requirements files as needed

### Scaling Options
- **Horizontal scaling** - Add more backend instances
- **Vertical scaling** - Upgrade instance size
- **Load balancing** - Distribute requests across instances
- **Caching layer** - Add Redis for frequent queries

### Performance Optimization
- **API response caching**
- **Request batching**
- **Async processing**
- **Database indexing**
- **CDN for static assets**

---

## 📞 Support

### Getting Help
- **Backend issues** - Check cloud provider docs
- **Frontend issues** - Streamlit Community Forum
- **API issues** - OpenAI Support
- **General issues** - GitHub Issues

### Useful Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DigitalOcean App Platform](https://docs.digitalocean.com/products/app-platform/)
- [Heroku Python Support](https://devcenter.heroku.com/articles/python-support)

---

*This hybrid architecture provides the best of both worlds: free frontend hosting with full-featured backend capabilities!*