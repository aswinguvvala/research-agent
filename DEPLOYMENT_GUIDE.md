# Deployment Guide: Lightweight AI Research Agent

Comprehensive guide for deploying the AI Research Agent across different platforms with memory and cost optimization.

## 🎯 Quick Start: Free Tier Deployment

### Streamlit Cloud (Recommended for UI)
**Cost:** Free  
**Memory:** ~1GB  
**Pros:** Easy deployment, managed infrastructure, great for frontend  
**Cons:** Memory constraints, 10min timeout, no persistent storage

**Steps:**
1. Fork/clone this repository
2. Create Streamlit Cloud account
3. Connect your GitHub repository
4. Add OpenAI API key to Secrets:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```
5. Deploy using `streamlit_app.py`

**Required files:**
- `streamlit_app.py` (main dashboard)
- `lightweight_research_system.py` (core system)
- `requirements-lightweight.txt` (dependencies)
- `.streamlit/config.toml` (configuration)

---

## 💡 Platform Comparison

| Platform | Memory | Cost/Month | Pros | Cons | Best For |
|----------|---------|------------|------|------|----------|
| **Streamlit Cloud** | 1GB | Free | Easy deployment, managed | Memory limits, no persistence | Frontend/Demo |
| **Heroku Basic** | 512MB | $7 | Simple, reliable | Small memory, expensive | Small apps |
| **Railway** | 1GB | $5 | Good free tier, simple | Limited resources | Development |
| **DigitalOcean** | 2GB | $12 | Full control, good value | Requires setup | Production |
| **AWS EC2 t3.small** | 2GB | $15 | Scalable, AWS ecosystem | Complex setup | Enterprise |
| **Google Cloud Run** | 2GB | Pay-per-use | Serverless, scales to zero | Complex pricing | Variable workload |

---

## 🚀 Deployment Options

### Option 1: Lightweight Version (Free Tier Compatible)

**Memory Usage:** ~800MB  
**Deployment:** Streamlit Cloud, Heroku Basic

**Features:**
- 3 specialized agents (Academic, Technical, Business)
- In-memory storage (no ChromaDB)
- GPT-4o-mini for cost optimization
- Basic web interface

**Setup:**
```bash
# Use lightweight requirements
pip install -r requirements-lightweight.txt

# Run locally
streamlit run streamlit_app.py

# Deploy to Streamlit Cloud
# Just connect your repo and add API key to secrets
```

### Option 2: Full System (Paid Hosting)

**Memory Usage:** ~2GB  
**Deployment:** DigitalOcean, AWS EC2 t3.small

**Features:**
- All 6 specialized agents
- ChromaDB vector storage
- Advanced reasoning engine
- Full API integration

**Setup:**
```bash
# Use full requirements
pip install -r requirements.txt

# Run full system
python quick_start.py
```

### Option 3: Hybrid Architecture

**Frontend:** Streamlit Cloud (Free)  
**Backend:** Paid cloud service

**Benefits:**
- Free frontend hosting
- Scalable backend processing
- Best of both worlds

---

## ⚙️ Configuration Files

### 1. Streamlit Configuration (`.streamlit/config.toml`)
```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
maxUploadSize = 50
maxMessageSize = 50

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
```

### 2. Environment Variables
```bash
# Required
OPENAI_API_KEY="your-openai-api-key"

# Optional (for full version)
SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-key"
NEWS_API_KEY="your-news-api-key"
```

### 3. Secrets (Streamlit Cloud)
```toml
OPENAI_API_KEY = "your-api-key-here"

[research_config]
max_agents = 3
timeout_seconds = 30
memory_limit_items = 50
```

---

## 🔧 Platform-Specific Instructions

### Streamlit Cloud
1. **Repository Setup:**
   - Ensure `streamlit_app.py` is in root directory
   - Add `requirements-lightweight.txt`
   - Include `.streamlit/` folder with config

2. **Deployment:**
   - Connect GitHub repo to Streamlit Cloud
   - Select `streamlit_app.py` as main file
   - Add API key in App Secrets

3. **Optimization:**
   - Use caching for expensive operations
   - Limit concurrent users
   - Implement session timeouts

### Heroku
1. **Setup:**
   ```bash
   # Create Procfile
   echo "web: streamlit run streamlit_app.py --server.port $PORT" > Procfile
   
   # Deploy
   heroku create your-app-name
   git push heroku main
   ```

2. **Environment Variables:**
   ```bash
   heroku config:set OPENAI_API_KEY="your-api-key"
   ```

### DigitalOcean Droplet
1. **Server Setup:**
   ```bash
   # Create 2GB droplet with Ubuntu
   # Install Python and dependencies
   sudo apt update && sudo apt install python3-pip
   pip3 install -r requirements-lightweight.txt
   
   # Run with screen or systemd
   screen -S research-agent
   streamlit run streamlit_app.py --server.address 0.0.0.0
   ```

2. **Domain Setup:**
   - Point domain to droplet IP
   - Use nginx as reverse proxy
   - Enable HTTPS with Let's Encrypt

### Docker Deployment
```bash
# Build lightweight version
docker build -f Dockerfile.lightweight -t research-agent:lightweight .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY="your-key" research-agent:lightweight

# Deploy to cloud container services
# (Google Cloud Run, AWS Fargate, etc.)
```

---

## 💰 Cost Analysis

### Research Session Costs (GPT-4o-mini)
- **Quick query (30s):** $0.001 - $0.005
- **Standard research (2min):** $0.01 - $0.03
- **Comprehensive analysis (5min):** $0.03 - $0.08

### Monthly Cost Estimates
| Usage Level | Sessions/Day | Monthly Cost |
|------------|--------------|--------------|
| Light | 5 | $1.50 - $4.50 |
| Medium | 20 | $6 - $18 |
| Heavy | 50 | $15 - $45 |

### Platform + API Costs
- **Streamlit Cloud + GPT-4o-mini:** $0 + API costs
- **Heroku Basic + GPT-4o-mini:** $7 + API costs
- **DigitalOcean + GPT-4o-mini:** $12 + API costs

---

## 🔍 Memory Optimization Strategies

### 1. Agent Reduction
```python
# Full system: 6 agents (~300MB)
agents = ["academic", "technical", "business", "social", "fact_checker", "synthesis"]

# Lightweight: 3 agents (~150MB)
agents = ["academic", "technical", "business"]
```

### 2. Storage Optimization
```python
# Full system: ChromaDB (~500MB)
vector_db = ChromaDB()

# Lightweight: In-memory (~50MB)
memory_store = SimpleMemoryStore(max_items=50)
```

### 3. Model Selection
```python
# Standard: Various models
models = ["gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002"]

# Lightweight: GPT-4o-mini only
model = "gpt-4o-mini"  # Cheapest and most efficient
```

---

## 🚨 Troubleshooting

### Memory Issues
**Problem:** App crashes due to memory limits  
**Solution:**
- Reduce max_items in SimpleMemoryStore
- Limit concurrent requests
- Use pagination for results
- Clear session state regularly

### API Rate Limits
**Problem:** OpenAI API rate limit exceeded  
**Solution:**
- Implement request throttling
- Add retry logic with exponential backoff
- Cache frequent queries
- Use lower token limits

### Deployment Failures
**Problem:** App fails to start on cloud platform  
**Solution:**
- Check requirements.txt compatibility
- Verify API key configuration
- Review logs for specific errors
- Test locally first

### Performance Issues
**Problem:** Slow response times  
**Solution:**
- Enable Streamlit caching
- Reduce token limits
- Limit agent selection
- Implement async processing

---

## ✅ Production Checklist

### Security
- [ ] API keys in secure environment variables
- [ ] Enable HTTPS
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Enable CORS protection

### Performance
- [ ] Enable caching
- [ ] Set appropriate timeouts
- [ ] Monitor memory usage
- [ ] Implement error handling
- [ ] Add health checks

### Monitoring
- [ ] Track API usage and costs
- [ ] Monitor system performance
- [ ] Log errors and issues
- [ ] Set up alerting
- [ ] Plan for scaling

### User Experience
- [ ] Add loading indicators
- [ ] Implement proper error messages
- [ ] Test on different devices
- [ ] Optimize for mobile
- [ ] Add usage instructions

---

## 🎉 Success Metrics

### Technical Metrics
- **Response Time:** < 30 seconds per query
- **Memory Usage:** < 1GB for free tier
- **API Cost:** < $0.05 per research session
- **Uptime:** > 99% availability

### User Metrics
- **Query Success Rate:** > 95%
- **User Satisfaction:** Based on feedback
- **Session Duration:** 2-5 minutes average
- **Return Usage:** Repeat users

---

## 🔄 Updates and Maintenance

### Regular Tasks
1. **Monitor API usage and costs**
2. **Update dependencies monthly**
3. **Review and optimize queries**
4. **Backup important data**
5. **Test new features in staging**

### Scaling Considerations
- **Traffic growth:** Consider paid hosting
- **Feature expansion:** Evaluate memory requirements
- **Cost optimization:** Regular cost reviews
- **Performance tuning:** Monitor and optimize bottlenecks

---

## 📞 Support and Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [DigitalOcean Tutorials](https://www.digitalocean.com/community/tutorials)

### Community
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/your-repo/issues)
- [Discord/Slack channels]

### Professional Support
- Streamlit Cloud Support
- OpenAI API Support
- Cloud provider support

---

*Last updated: December 2024*