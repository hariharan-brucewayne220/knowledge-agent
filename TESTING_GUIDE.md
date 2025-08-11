# 🧪 Content Ingestion Pipeline - Step-by-Step Testing Guide

Follow this guide to test all the pipeline features we've implemented.

## ✅ Prerequisites Completed

The standalone test already confirmed:
- ✅ Pipeline modules import correctly
- ✅ Pipeline can be created and configured
- ✅ File watcher works (polling mode)
- ✅ Configuration system works
- ✅ Directory monitoring is set up

## 🚀 Step 1: Start the Server

**In Terminal 1 (Keep this running):**
```bash
# Navigate to project directory
cd /mnt/d/claude-projects/knowagent-research

# Start the server
myenv/Scripts/python.exe app.py
```

You should see:
```
✅ Content Ingestion Pipeline API enabled
INFO:     Started server process [...]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open - it's your server!**

## 🧪 Step 2: Test API Endpoints

**In Terminal 2 (New terminal):**
```bash
# Test all API endpoints
myenv/Scripts/python.exe test_api_endpoints.py
```

This will test:
- ✅ Server health check
- ✅ Pipeline status
- ✅ Start/stop pipeline
- ✅ Add/list sources
- ✅ File detection
- ✅ Recent items

## 📁 Step 3: Test Real-Time File Detection

**While server is running:**

### Test 1: Manual File Drop
```bash
# Copy a PDF to watched folder
copy any_pdf_file.pdf watched_pdfs/
```

**Expected Result:** 
- Server console shows: `[PROCESSING] Processing PDF: filename.pdf`
- File automatically processed and added to knowledge base

### Test 2: API File Drop
```bash
# Create test directory
mkdir api_test_pdfs

# Copy PDF there
copy any_pdf_file.pdf api_test_pdfs/

# Add source via API
curl -X POST http://localhost:8000/api/pipeline/sources/pdf \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_source", 
    "path_or_url": "api_test_pdfs/",
    "enabled": true
  }'
```

## 🔍 Step 4: Monitor Pipeline Status

**Check pipeline status:**
```bash
# Via curl
curl http://localhost:8000/api/pipeline/status

# Via browser
# Open: http://localhost:8000/api/pipeline/status
```

**Expected Response:**
```json
{
  "is_running": true,
  "monitored_sources": 3,
  "queue_size": 0,
  "processed_items": 2,
  "stats": {
    "total_discovered": 2,
    "total_processed": 2,
    "success_rate": 100.0
  }
}
```

## 📊 Step 5: Test All API Endpoints

### Core Pipeline Control
```bash
# Start pipeline
curl -X POST http://localhost:8000/api/pipeline/start

# Stop pipeline  
curl -X POST http://localhost:8000/api/pipeline/stop

# Get status
curl http://localhost:8000/api/pipeline/status
```

### Source Management
```bash
# List sources
curl http://localhost:8000/api/pipeline/sources

# Add PDF source
curl -X POST http://localhost:8000/api/pipeline/sources/pdf \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_papers",
    "path_or_url": "research/",
    "enabled": true,
    "recursive": true
  }'

# Remove source
curl -X DELETE http://localhost:8000/api/pipeline/sources/research_papers
```

### Monitoring
```bash
# Recent items
curl http://localhost:8000/api/pipeline/recent-items

# Configuration
curl http://localhost:8000/api/pipeline/config

# Health check
curl http://localhost:8000/api/pipeline/health
```

## 🎥 Step 6: Test YouTube Integration (Optional)

**Note:** Requires YouTube API key

### Add YouTube API Key
```bash
# Edit pipeline_config.json
# Add: "youtube_api_key": "your-api-key-here"
```

### Add YouTube Source
```bash
curl -X POST http://localhost:8000/api/pipeline/sources/youtube \
  -H "Content-Type: application/json" \
  -d '{
    "name": "educational_videos",
    "url": "https://www.youtube.com/playlist?list=PLAYLIST_ID",
    "enabled": true,
    "check_interval": 3600
  }'
```

## 🧪 Step 7: Integration with Existing System

### Test with Existing Research System
```bash
# Query the system (should include auto-processed content)
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what does the latest document discuss",
    "pdf_files": [],
    "youtube_urls": []
  }'
```

**Expected:** New PDFs should appear in search results automatically.

## 📈 Step 8: Performance Testing

### Batch File Processing
```bash
# Copy multiple PDFs to watched folder
copy *.pdf watched_pdfs/

# Monitor processing
curl http://localhost:8000/api/pipeline/recent-items
```

### Stress Test
```bash
# Create 10 test files
for i in {1..10}; do
  echo "Test document $i" > "watched_pdfs/test_$i.pdf"
done

# Check processing queue
curl http://localhost:8000/api/pipeline/status
```

## ✅ Success Criteria

After testing, you should see:

### ✅ Basic Functionality
- [ ] Server starts without errors
- [ ] Pipeline can be started/stopped via API
- [ ] Files are detected within 30 seconds
- [ ] Processing completes successfully
- [ ] Processed files appear in knowledge base

### ✅ API Functionality  
- [ ] All endpoints return 200 OK
- [ ] Sources can be added/removed
- [ ] Status shows correct information
- [ ] Recent items list updates

### ✅ Real-Time Features
- [ ] New PDFs detected automatically
- [ ] Processing happens in background
- [ ] Multiple files processed concurrently
- [ ] Failed items are retried

### ✅ Integration
- [ ] Processed content appears in research queries
- [ ] Source attribution works
- [ ] No conflicts with existing system

## 🐛 Troubleshooting

### Pipeline Not Starting
```bash
# Check server logs for errors
# Look for import errors or missing dependencies
```

### Files Not Detected
```bash
# Check if directory exists and is configured
curl http://localhost:8000/api/pipeline/sources

# Check file permissions
# Ensure PDFs are > 1KB in size
```

### Processing Failures
```bash
# Check recent items for error messages
curl http://localhost:8000/api/pipeline/recent-items?limit=50

# Check server console for detailed error logs
```

### API Errors
```bash
# Verify server is running on correct port
netstat -an | findstr 8000

# Check firewall/antivirus blocking connections
```

## 🎯 Expected Results

**After successful testing:**
- ✅ **Real-time PDF processing** working
- ✅ **API control** of pipeline
- ✅ **Background processing** active
- ✅ **Source management** functional
- ✅ **Integration** with existing research system
- ✅ **Automatic knowledge base updates**

## 🚀 Ready for Production!

Once all tests pass, your **Real-Time Content Ingestion Pipeline** is ready for production use. You now have:

- **Automatic content discovery** from multiple sources
- **Background processing** without user intervention  
- **RESTful API control** for programmatic management
- **Real-time knowledge base updates**
- **Professional monitoring and health checks**

**Your research system has evolved from static to dynamic!** 🎉