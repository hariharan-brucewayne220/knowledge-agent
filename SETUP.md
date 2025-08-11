# KnowledgeAgent Setup Guide

## Environment Configuration

### 1. Copy Environment Template
```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit the `.env` file and add your API keys:

```env
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-your-actual-openai-api-key-here

# YouTube Data API v3 Configuration  
YOUTUBE_API_KEY=your-actual-youtube-api-key-here

# Optional Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
MAX_CONCURRENT_PROCESSORS=3
YOUTUBE_CHECK_INTERVAL_HOURS=24
```

### 3. API Key Setup Instructions

#### OpenAI API Key:
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create new secret key
3. Copy the key (starts with `sk-proj-`)
4. Add to `.env` file

#### YouTube Data API Key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project or select existing
3. Enable "YouTube Data API v3"
4. Go to Credentials → Create credentials → API Key
5. Copy the API key
6. Add to `.env` file

### 4. Configure YouTube Monitoring

Edit `pipeline_config.json` to add channels/playlists:

```json
{
  "youtube_monitoring": {
    "enabled": true,
    "sources": {
      "mit_ai": {
        "type": "youtube_channel",
        "channel_id": "UCEBb1b_L6zDS3xTUrIALZOw",
        "enabled": true,
        "keywords_filter": ["AI", "machine learning"],
        "min_duration_seconds": 300
      }
    }
  }
}
```

### 5. Start the System

```bash
# Start with auto-processing
myenv/Scripts/python.exe app.py

# Or use the startup script
myenv/Scripts/python.exe start_pipeline_server.py
```

## Security Notes

- ⚠️ **Never commit `.env` file to git**
- ✅ The `.env` file is already in `.gitignore`
- ✅ Use `.env.example` as a template for others
- ✅ API keys are loaded automatically from environment variables

## Testing the Setup

1. **PDF Auto-Processing**: Drop a PDF in `watched_pdfs/` folder
2. **YouTube Monitoring**: Restart server to trigger YouTube check
3. **Research Queries**: Use the web interface to ask questions

## Troubleshooting

- **No OpenAI responses**: Check `OPENAI_API_KEY` in `.env`
- **YouTube monitoring disabled**: Check `YOUTUBE_API_KEY` in `.env`  
- **Server not starting**: Check for syntax errors in configuration files