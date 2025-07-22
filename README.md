# KnowledgeAgent Research Assistant 🧠

An intelligent research system that combines **multi-modal content processing**, **enhanced NER-based routing**, and **anti-hallucination verification** to provide trustworthy, source-attributed research assistance.

## 🎯 What Makes This Unique

Unlike ChatGPT/Claude, this system provides:
- **Zero Hallucination** - All answers cite exact sources (PDF pages, video timestamps)
- **Multi-Modal Intelligence** - Processes PDFs AND YouTube videos seamlessly  
- **Enhanced Content Routing** - Contextually understands queries (distinguishes "solar system" vs "solar panels")
- **Research-Grade Provenance** - Perfect for academic/professional use where attribution matters

## ⚡ Key Features

### 🔍 Intelligent Content Routing
- **Enhanced NER-based fuzzy search** with contextual awareness
- **Query intent analysis** - understands theoretical vs practical questions
- **Cross-domain intelligence** - finds connections between different knowledge areas
- **Adaptive result ranking** based on query complexity

### 📚 Multi-Modal Processing  
- **PDF Processing** - Extract, chunk, and embed academic papers
- **YouTube Integration** - Auto-transcribe videos with timestamp preservation
- **Unified Content Store** - Single interface for all content types
- **Vector Search** - Semantic similarity matching across all sources

### 🛡️ Anti-Hallucination Architecture
- **Source Provenance** - Every answer shows exact page/timestamp
- **Content Verification** - Responses verified against stored content
- **Citation Generation** - Academic-grade source attribution
- **Confidence Scoring** - Transparency about answer reliability

### 🎨 Advanced UI Features
- **Research Dashboard** - Visual content organization
- **Topic Classification** - Dynamic content categorization  
- **Interactive Source Explorer** - Click to view original content
- **Real-time Processing** - Background content ingestion

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 FastAPI Backend                     │
├─────────────────────────────────────────────────────┤
│  Enhanced Research Executor                         │
│  ├─ Query Intent Analysis                          │
│  ├─ Enhanced NER Router                            │
│  ├─ Multi-Source Synthesis                        │
│  └─ Source Verification                           │
├─────────────────────────────────────────────────────┤
│  Content Processing Pipeline                        │
│  ├─ PDF Agent (PyMuPDF + OpenAI Embeddings)      │
│  ├─ YouTube Agent (yt-dlp + Whisper)             │
│  ├─ Topic Classifier (Dynamic NER)                │
│  └─ Vector Store (ChromaDB)                       │
├─────────────────────────────────────────────────────┤
│  Storage Layer                                      │
│  ├─ UnifiedContentStore                           │
│  ├─ Vector Database (ChromaDB)                    │
│  ├─ Topic Classifications                         │
│  └─ Content Index                                 │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install -r requirements.txt

# Optional: Redis for caching
redis-server
```

### Environment Setup
```bash
# Required for OpenAI features
export OPENAI_API_KEY="your-openai-key"

# Optional: YouTube API for metadata
export YOUTUBE_API_KEY="your-youtube-key" 
```

### Launch System
```bash
# Start the server
python app.py

# Access the interface
open http://localhost:8000
```

## 📖 Usage Examples

### Research Query
```
Query: "What does quantum theory say about dark matter?"
Result: 
├─ Sources: Dark Matter PDF (Page 1) + Quantum Physics Video (12:34)
├─ Confidence: 0.87
└─ Provenance: [Exact citations with page numbers/timestamps]
```

### Content Upload
- **Drag & drop PDFs** → Auto-processed and indexed
- **Paste YouTube URLs** → Auto-transcribed and integrated
- **Background processing** → Real-time status updates

### Smart Routing Examples
- `"solar system exploration"` → Routes to space/astronomy content
- `"solar panel efficiency"` → Routes to energy technology content  
- `"compare renewable energy storage"` → Multi-PDF synthesis

## 🔧 Technical Implementation

### Core Components

#### Enhanced NER Router
```python
class EnhancedNERFuzzyRouter:
    def route_query(self, query: str):
        # 1. Extract query terms with NER
        # 2. Analyze query intent (theoretical/practical)
        # 3. Score documents with contextual awareness
        # 4. Apply domain-specific boosting
        # 5. Return ranked, relevant sources
```

#### Multi-Modal Content Store  
```python
class UnifiedContentStore:
    def add_pdf_content(self, file_path, chunks, metadata)
    def add_youtube_content(self, url, segments, metadata)  
    def search_content(self, query, max_results)
    def get_all_content(self) -> List[UnifiedContentItem]
```

#### Research Executor
```python
class EnhancedResearchExecutor:
    async def execute_research_query(self, query, sources):
        # 1. Route query to relevant content
        # 2. Extract and verify information  
        # 3. Synthesize multi-source response
        # 4. Generate source citations
        # 5. Return verified answer with provenance
```

## 📊 Content Statistics

The system currently processes:
- **Academic PDFs** - Research papers, technical documents
- **Educational Videos** - Lectures, tutorials, documentaries  
- **Cross-Domain Topics** - Physics, Technology, Environment, etc.

Sample content includes:
- Dark Matter & Quantum Physics research
- Renewable Energy & Carbon Capture studies  
- Space Exploration & Planetary Science videos
- Technical documentation and academic papers

## 🎛️ API Endpoints

### Core Research
- `POST /api/research` - Execute research query
- `POST /api/upload-pdf` - Upload PDF document
- `POST /api/parse-input` - Parse user input (text + URLs)

### Content Management  
- `GET /api/simple-content` - List all content
- `POST /api/search-content` - Search stored content
- `GET /api/topics` - Get topic classification hierarchy

### System Status
- `GET /api/status` - System health and statistics
- `GET /api/export` - Export research library
- `POST /api/delete-all-data` - Reset system

## 🧪 Advanced Features

### Research Connections Analysis
- **Contradiction Detection** - Find conflicting information
- **Confirmation Analysis** - Identify supporting evidence  
- **Extension Discovery** - Locate complementary research
- **Knowledge Gap Identification** - Highlight missing information

### Dynamic Topic Classification
- **Automatic Categorization** - ML-based content classification
- **Cross-Topic Discovery** - Find multi-domain content
- **Research Hierarchy** - Organized knowledge structure  
- **Confidence Scoring** - Classification reliability metrics

### Enhanced Search Capabilities
- **Semantic Search** - Vector similarity matching
- **Contextual Routing** - Query intent understanding
- **Multi-Modal Fusion** - Combined PDF + video results
- **Relevance Ranking** - Intelligent result ordering

## 🔬 Research Applications

### Academic Research
- Literature review with source verification
- Cross-reference analysis across papers
- Citation generation with page numbers
- Research gap identification

### Corporate R&D  
- Technical documentation analysis
- Product research with video integration
- Competitive analysis with source tracking
- Innovation opportunity detection

### Educational Use
- Curriculum development with multi-modal sources
- Student research assistance with verified sources
- Lecture preparation with content synthesis
- Academic integrity through source attribution

## 🛠️ Development Roadmap

### Current Status ✅
- [x] Multi-modal content processing
- [x] Enhanced NER-based routing  
- [x] Anti-hallucination verification
- [x] Source provenance system
- [x] Dynamic topic classification
- [x] Research dashboard UI

### Next Features 🚧
- [ ] Real-time content ingestion pipeline
- [ ] Collaborative research workspaces
- [ ] Advanced analytics dashboard
- [ ] Export to academic formats (LaTeX, BibTeX)
- [ ] API integrations (Zotero, Mendeley)

### Future Vision 🔮
- [ ] Multi-language support
- [ ] Voice interface integration
- [ ] Real-time collaboration features
- [ ] Advanced AI reasoning chains
- [ ] Custom domain specialization

## 🏆 Competitive Advantages

| Feature | KnowledgeAgent | ChatGPT/Claude |
|---------|----------------|----------------|
| **Source Attribution** | ✅ Exact page/timestamp | ❌ Generic references |
| **Multi-Modal Processing** | ✅ PDF + Video unified | ❌ Limited integration |
| **Zero Hallucination** | ✅ Verified responses | ❌ May generate false info |
| **Domain Specialization** | ✅ Customizable knowledge | ❌ Generic training |
| **Research Provenance** | ✅ Academic-grade citations | ❌ Unreliable attribution |

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/knowagent-research.git

# Install dependencies  
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development server
python app.py --reload
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **KnowAgent Paper** - Original research framework
- **OpenAI** - Embeddings and language models  
- **ChromaDB** - Vector database implementation
- **FastAPI** - Modern web framework
- **Hugging Face** - NLP models and tools

## 📞 Support

- **Documentation**: [Full docs](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/knowagent-research/issues)  
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/knowagent-research/discussions)

---

**Built with ❤️ for researchers who demand accuracy and attribution in their AI-assisted work.**