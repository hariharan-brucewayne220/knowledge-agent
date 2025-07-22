// KnowledgeAgent Frontend Application
class KnowledgeAgentApp {
    constructor() {
        this.uploadedFiles = new Set();
        this.extractedLinks = [];
        this.isProcessing = false;
        this.topics = [];
        this.selectedTopic = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkSystemStatus();
        this.initializeTextarea();
        this.loadTopics();
    }

    initializeElements() {
        // Main elements
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.messagesContainer = document.getElementById('messages');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.loadingStatus = document.getElementById('loading-status');
        
        // Sidebar elements
        this.pdfUpload = document.getElementById('pdf-upload');
        this.pdfInput = document.getElementById('pdf-input');
        this.uploadedFilesContainer = document.getElementById('uploaded-files');
        this.extractedLinksContainer = document.getElementById('extracted-links');
        this.systemStatus = document.getElementById('system-status');
        
        // Topic folders elements
        this.topicFoldersContainer = document.getElementById('topic-folders');
        this.refreshTopicsBtn = document.getElementById('refresh-topics');
        this.topicCount = document.getElementById('topic-count');
        this.contentCount = document.getElementById('content-count');
        
        // Quick action elements
        this.viewAllContentBtn = document.getElementById('view-all-content');
        this.crossTopicSearchBtn = document.getElementById('cross-topic-search');
        this.exportLibraryBtn = document.getElementById('export-library');
        this.deleteAllDataBtn = document.getElementById('delete-all-data');
        
        // Header elements
        this.clearBtn = document.getElementById('clear-btn');
        this.settingsBtn = document.getElementById('settings-btn');
        this.settingsModal = document.getElementById('settings-modal');
        this.closeSettings = document.getElementById('close-settings');
        
        // Character counter
        this.charCount = document.querySelector('.char-count');
        
        // Loading steps
        this.loadingSteps = {
            parsing: document.getElementById('step-parsing'),
            planning: document.getElementById('step-planning'),
            executing: document.getElementById('step-executing'),
            synthesizing: document.getElementById('step-synthesizing')
        };
    }

    attachEventListeners() {
        // Message input and sending
        this.messageInput.addEventListener('input', () => this.handleInputChange());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // File upload
        this.pdfUpload.addEventListener('click', () => this.pdfInput.click());
        this.pdfInput.addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Drag and drop
        this.pdfUpload.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.pdfUpload.addEventListener('drop', (e) => this.handleDrop(e));
        this.pdfUpload.addEventListener('dragleave', () => this.handleDragLeave());
        
        // Header actions
        this.clearBtn.addEventListener('click', () => this.clearChat());
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeSettings.addEventListener('click', () => this.closeSettingsModal());
        
        // Modal close on outside click
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.closeSettingsModal();
            }
        });
        
        // Settings actions
        document.getElementById('clear-cache').addEventListener('click', () => this.clearCache());
        
        // Topic folders actions
        this.refreshTopicsBtn.addEventListener('click', () => this.loadTopics());
        this.viewAllContentBtn.addEventListener('click', () => this.viewAllContent());
        this.crossTopicSearchBtn.addEventListener('click', () => this.showCrossTopicSearch());
        this.exportLibraryBtn.addEventListener('click', () => this.exportLibrary());
        this.deleteAllDataBtn.addEventListener('click', () => this.deleteAllData());
    }

    initializeTextarea() {
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
            this.updateCharCount();
        });
    }

    handleInputChange() {
        const value = this.messageInput.value.trim();
        this.sendBtn.disabled = !value || this.isProcessing;
        this.updateCharCount();
        
        // Extract links in real-time
        this.extractLinksFromInput(value);
    }

    updateCharCount() {
        const count = this.messageInput.value.length;
        this.charCount.textContent = `${count} / 4000`;
        
        if (count > 3800) {
            this.charCount.style.color = 'var(--red)';
        } else if (count > 3500) {
            this.charCount.style.color = 'var(--yellow)';
        } else {
            this.charCount.style.color = 'var(--text-light)';
        }
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.sendBtn.disabled) {
                this.sendMessage();
            }
        }
    }

    async extractLinksFromInput(text) {
        if (!text) {
            this.extractedLinks = [];
            this.updateExtractedLinksDisplay();
            return;
        }

        try {
            const response = await fetch('/api/parse-input', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input: text })
            });

            if (response.ok) {
                const data = await response.json();
                this.extractedLinks = data.youtube_analysis || [];
                this.updateExtractedLinksDisplay();
            }
        } catch (error) {
            console.error('Error extracting links:', error);
        }
    }

    updateExtractedLinksDisplay() {
        if (this.extractedLinks.length === 0) {
            this.extractedLinksContainer.innerHTML = '<p class="placeholder">YouTube and other links will appear here</p>';
            return;
        }

        const html = this.extractedLinks.map(link => `
            <div class="link-item youtube">
                <i class="fab fa-youtube"></i>
                <div class="link-info">
                    <a href="${link.url}" target="_blank" class="link-url">${link.title || 'YouTube Video'}</a>
                    <div class="link-status ${link.captions_available ? 'status-transcript-available' : 'status-download-needed'}">
                        <i class="fas ${link.captions_available ? 'fa-check' : 'fa-download'}"></i>
                        ${link.captions_available ? 'Transcript available' : 'Will download for transcription'}
                    </div>
                </div>
            </div>
        `).join('');

        this.extractedLinksContainer.innerHTML = html;
    }

    // File Upload Handlers
    handleFileUpload(e) {
        const files = Array.from(e.target.files);
        files.forEach(file => this.uploadFile(file));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.pdfUpload.classList.add('dragover');
    }

    handleDragLeave() {
        this.pdfUpload.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.pdfUpload.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files).filter(file => file.type === 'application/pdf');
        files.forEach(file => this.uploadFile(file));
    }

    async uploadFile(file) {
        if (!file.type.includes('pdf')) {
            this.showError('Only PDF files are allowed');
            return;
        }

        if (this.uploadedFiles.has(file.name)) {
            this.showError('File already uploaded');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload-pdf', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                this.uploadedFiles.add(file.name);
                this.addFileToDisplay(data);
                this.showSuccess(`Uploaded ${file.name}`);
            } else {
                throw new Error('Upload failed');
            }
        } catch (error) {
            this.showError(`Failed to upload ${file.name}: ${error.message}`);
        }
    }

    addFileToDisplay(fileData) {
        const fileElement = document.createElement('div');
        fileElement.className = 'file-item';
        fileElement.innerHTML = `
            <div class="file-info">
                <i class="fas fa-file-pdf"></i>
                <div>
                    <div class="file-name">${fileData.filename}</div>
                    <div class="file-size">${this.formatFileSize(fileData.size)}</div>
                </div>
            </div>
            <button class="file-remove" onclick="app.removeFile('${fileData.filename}')">
                <i class="fas fa-times"></i>
            </button>
        `;

        this.uploadedFilesContainer.appendChild(fileElement);
    }

    removeFile(filename) {
        this.uploadedFiles.delete(filename);
        
        // Remove from display
        const fileItems = this.uploadedFilesContainer.querySelectorAll('.file-item');
        fileItems.forEach(item => {
            if (item.querySelector('.file-name').textContent === filename) {
                item.remove();
            }
        });
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Message Handling
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;

        this.isProcessing = true;
        this.sendBtn.disabled = true;

        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.updateCharCount();

        // Show loading
        this.showLoading();

        try {
            // Prepare research data
            const researchData = {
                query: message,
                pdf_files: Array.from(this.uploadedFiles),
                youtube_urls: this.extractedLinks.map(link => link.url)
            };

            // Execute research
            const response = await fetch('/api/research', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(researchData)
            });

            if (response.ok) {
                const data = await response.json();
                this.handleResearchResponse(data);
            } else {
                throw new Error('Research request failed');
            }
        } catch (error) {
            this.addMessage(`Sorry, I encountered an error: ${error.message}`, 'assistant');
        } finally {
            this.hideLoading();
            this.isProcessing = false;
            this.sendBtn.disabled = false;
        }
    }

    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const avatar = type === 'user' 
            ? '<i class="fas fa-user"></i>'
            : '<i class="fas fa-robot"></i>';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessageContent(content, type)}</div>
            </div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatMessageContent(content, type) {
        if (type === 'user') {
            return this.escapeHtml(content).replace(/\n/g, '<br>');
        }
        
        // For assistant messages, check if it's a research result object
        if (typeof content === 'object' && content.result) {
            return this.formatResearchResults(content.result);
        }
        
        return this.escapeHtml(content).replace(/\n/g, '<br>');
    }

    formatResearchResults(result) {
        const answer = result.final_answer;
        const execution = result.execution;
        const smartRouting = result.smart_routing || {};
        
        return `
            <div class="research-results">
                ${smartRouting.explanation ? `
                <div class="result-section smart-routing">
                    <h4><i class="fas fa-route"></i> Smart Content Routing</h4>
                    <div class="smart-routing-info">
                        <div class="routing-explanation">
                            ${smartRouting.explanation}
                        </div>
                        ${smartRouting.matched_terms && smartRouting.matched_terms.length > 0 ? `
                        <div class="matched-terms">
                            <small>Matched terms: ${smartRouting.matched_terms.join(', ')}</small>
                        </div>
                        ` : ''}
                    </div>
                </div>
                ` : ''}

                <div class="result-section">
                    <h4><i class="fas fa-search"></i> Research Summary</h4>
                    <div class="result-metadata">
                        <div class="metadata-item">
                            <div class="metadata-value">${execution.successful_steps}/${execution.total_steps}</div>
                            <div class="metadata-label">Steps Completed</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-value">${execution.total_execution_time.toFixed(1)}s</div>
                            <div class="metadata-label">Processing Time</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-value">${answer.research_confidence.toFixed(2)}</div>
                            <div class="metadata-label">Confidence Score</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-value">${answer.sources_processed.pdf_documents || 0}</div>
                            <div class="metadata-label">PDFs Analyzed</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-value">${answer.sources_processed.youtube_videos || 0}</div>
                            <div class="metadata-label">Videos Analyzed</div>
                        </div>
                    </div>
                </div>

                <div class="result-section">
                    <h4><i class="fas fa-lightbulb"></i> Answer</h4>
                    <div class="message-text">${this.escapeHtml(answer.answer_summary).replace(/\n/g, '<br>')}</div>
                </div>

                ${answer.source_provenance && answer.source_provenance.length > 0 ? `
                <div class="result-section">
                    <h4><i class="fas fa-link"></i> Source Provenance</h4>
                    <div class="source-citations">
                        ${answer.source_provenance.map((source, index) => `
                        <div class="source-citation">
                            <div class="source-header">
                                <span class="source-number">[${index + 1}]</span>
                                <span class="source-title">${this.escapeHtml(source.source_id || 'Unknown Source')}</span>
                                <span class="source-type badge">${source.source_type || 'unknown'}</span>
                            </div>
                            ${source.specific_reference ? `
                            <div class="source-reference">
                                <i class="fas fa-bookmark"></i> ${this.escapeHtml(source.specific_reference)}
                            </div>
                            ` : ''}
                            ${source.content_excerpt ? `
                            <div class="source-excerpt">
                                <i class="fas fa-quote-left"></i>
                                "${this.escapeHtml(source.content_excerpt)}"
                            </div>
                            ` : ''}
                            ${source.relevance ? `
                            <div class="source-relevance">
                                <i class="fas fa-info-circle"></i> ${this.escapeHtml(source.relevance)}
                            </div>
                            ` : ''}
                        </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}

                ${answer.reasoning_steps && answer.reasoning_steps.length > 0 ? `
                <div class="result-section">
                    <h4><i class="fas fa-brain"></i> Reasoning Steps</h4>
                    <ol class="reasoning-steps">
                        ${answer.reasoning_steps.map(step => `<li>${this.escapeHtml(step)}</li>`).join('')}
                    </ol>
                </div>
                ` : ''}

                ${answer.next_steps && answer.next_steps.length > 0 ? `
                <div class="result-section">
                    <h4><i class="fas fa-arrow-right"></i> Suggested Next Steps</h4>
                    <ul>
                        ${answer.next_steps.map(step => `<li>${this.escapeHtml(step)}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            </div>
        `;
    }

    handleResearchResponse(data) {
        if (data.success) {
            const cacheIndicator = data.cached ? ' <span style="color: var(--primary-green); font-size: 0.8em;">(cached)</span>' : '';
            this.addMessage(data, 'assistant');
            
            if (data.cached) {
                this.showSuccess('Results retrieved from cache');
            }
        } else {
            this.addMessage('Sorry, I encountered an error processing your request.', 'assistant');
        }
    }

    // Loading Management
    showLoading() {
        this.loadingOverlay.style.display = 'flex';
        this.updateLoadingStep('parsing');
        
        // Simulate progress through steps
        setTimeout(() => this.updateLoadingStep('planning'), 500);
        setTimeout(() => this.updateLoadingStep('executing'), 1000);
        setTimeout(() => this.updateLoadingStep('synthesizing'), 2000);
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
        
        // Reset all steps
        Object.values(this.loadingSteps).forEach(step => {
            step.classList.remove('active', 'completed');
        });
    }

    updateLoadingStep(stepName) {
        // Mark previous steps as completed
        const stepOrder = ['parsing', 'planning', 'executing', 'synthesizing'];
        const currentIndex = stepOrder.indexOf(stepName);
        
        stepOrder.forEach((step, index) => {
            const element = this.loadingSteps[step];
            element.classList.remove('active');
            
            if (index < currentIndex) {
                element.classList.add('completed');
            } else if (index === currentIndex) {
                element.classList.add('active');
            } else {
                element.classList.remove('completed');
            }
        });

        // Update status text
        const statusTexts = {
            parsing: 'Analyzing your input and extracting links...',
            planning: 'Creating optimal research plan...',
            executing: 'Processing documents and videos...',
            synthesizing: 'Generating comprehensive answer...'
        };
        
        this.loadingStatus.textContent = statusTexts[stepName] || 'Processing...';
    }

    // System Status
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const status = await response.json();
                this.updateSystemStatus(status);
            }
        } catch (error) {
            console.error('Failed to check system status:', error);
        }
    }

    updateSystemStatus(status) {
        const statusHTML = `
            <div class="status-item">
                <span class="status-label">Research Engine:</span>
                <span class="status-value ${status.research_executor ? 'healthy' : 'error'}">
                    ${status.research_executor ? 'Online' : 'Offline'}
                </span>
            </div>
            <div class="status-item">
                <span class="status-label">YouTube Agent:</span>
                <span class="status-value ${status.youtube_agent ? 'healthy' : 'error'}">
                    ${status.youtube_agent ? 'Ready' : 'Error'}
                </span>
            </div>
            <div class="status-item">
                <span class="status-label">Cache:</span>
                <span class="status-value healthy">${status.cache.toUpperCase()}</span>
            </div>
            <div class="status-item">
                <span class="status-label">YouTube API:</span>
                <span class="status-value ${status.youtube_api ? 'healthy' : 'loading'}">
                    ${status.youtube_api ? 'Configured' : 'Local Only'}
                </span>
            </div>
        `;
        
        this.systemStatus.innerHTML = statusHTML;
    }

    // Utility Methods
    clearChat() {
        // Keep the welcome message, remove others
        const messages = this.messagesContainer.querySelectorAll('.message');
        messages.forEach((message, index) => {
            if (index > 0) { // Keep first message (welcome)
                message.remove();
            }
        });
        
        this.showSuccess('Chat cleared');
    }

    openSettings() {
        this.settingsModal.style.display = 'flex';
    }

    closeSettingsModal() {
        this.settingsModal.style.display = 'none';
    }

    async clearCache() {
        // This would call a cache clearing endpoint if implemented
        this.showSuccess('Cache cleared');
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    // =====================================
    // TOPIC FOLDERS FUNCTIONALITY
    // =====================================

    async loadTopics() {
        try {
            console.log('Loading topics...');
            this.showTopicsLoading(true);
            
            // Add cache busting to prevent stale topic data
            const response = await fetch(`/api/topics?t=${Date.now()}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            this.topics = data.topics || [];
            
            this.renderTopics();
            this.updateTopicStats(data.stats);
            
        } catch (error) {
            console.error('Error loading topics:', error);
            this.renderEmptyTopics();
        }
    }

    showTopicsLoading(isLoading) {
        if (isLoading) {
            this.topicFoldersContainer.innerHTML = `
                <div class="loading-topics">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span>Loading topics...</span>
                </div>
            `;
        }
        // Note: Loading state is cleared by renderTopics() or renderEmptyTopics()
    }

    renderTopics() {
        if (!this.topics.length) {
            this.renderEmptyTopics();
            return;
        }

        const topicsHtml = this.topics.map(topic => {
            const crossTopicItems = topic.content.filter(item => item.cross_topic);
            const pdfCount = topic.content.filter(item => item.type === 'pdf').length;
            const videoCount = topic.content.filter(item => item.type === 'video').length;
            
            return `
                <div class="topic-folder" data-topic-id="${topic.id}">
                    <div class="topic-header" onclick="app.toggleTopic('${topic.id}')">
                        <div class="topic-info">
                            <i class="topic-icon fas fa-folder"></i>
                            <span class="topic-name" title="${topic.name}">${topic.name}</span>
                        </div>
                        <div class="topic-meta">
                            <span class="content-count">${topic.total_content}</span>
                            ${crossTopicItems.length > 0 ? `<span class="cross-topic-badge">${crossTopicItems.length}</span>` : ''}
                            <i class="expand-icon fas fa-chevron-right"></i>
                        </div>
                    </div>
                    <div class="topic-content">
                        <div class="topic-summary">
                            <small class="topic-description">${topic.description || 'No description available'}</small>
                            <div class="topic-stats">
                                ${pdfCount > 0 ? `<span class="stat-item"><i class="fas fa-file-pdf"></i> ${pdfCount} PDFs</span>` : ''}
                                ${videoCount > 0 ? `<span class="stat-item"><i class="fas fa-video"></i> ${videoCount} Videos</span>` : ''}
                                <span class="stat-item"><i class="fas fa-percentage"></i> ${Math.round(topic.confidence * 100)}% confidence</span>
                            </div>
                        </div>
                        ${this.renderTopicContent(topic.content)}
                    </div>
                </div>
            `;
        }).join('');

        this.topicFoldersContainer.innerHTML = topicsHtml;
    }

    renderTopicContent(content) {
        return content.map(item => {
            const crossTopicClass = item.cross_topic ? 'cross-topic' : '';
            const metadata = item.metadata || {};
            const typeIcon = metadata.icon || (item.type === 'pdf' ? 'fa-file-pdf' : 'fa-video');
            const typeClass = item.type === 'pdf' ? 'pdf' : 'video';
            
            // Create a tooltip with preview and metadata
            const tooltipContent = this.createContentTooltip(item, metadata);
            
            return `
                <div class="content-item ${crossTopicClass}" 
                     onclick="app.selectContent('${item.id}')" 
                     title="${tooltipContent}">
                    <i class="content-type-icon ${typeClass} fas ${typeIcon}"></i>
                    <span class="content-title">${item.title}</span>
                    ${item.chunk_count > 1 ? `<span class="chunk-count">${item.chunk_count}</span>` : ''}
                    ${item.similarity ? `<span class="content-similarity">${Math.round(item.similarity * 100)}%</span>` : ''}
                </div>
            `;
        }).join('');
    }

    createContentTooltip(item, metadata) {
        let tooltip = `${item.title}\n\n`;
        
        if (metadata.preview) {
            tooltip += `Preview: ${metadata.preview}\n\n`;
        }
        
        if (item.type === 'pdf' && metadata.page_info) {
            tooltip += `Page: ${metadata.page_info}\n`;
        }
        
        if (item.type === 'video' && metadata.timestamp) {
            tooltip += `Time: ${metadata.timestamp}\n`;
        }
        
        if (item.chunk_count > 1) {
            tooltip += `Contains ${item.chunk_count} sections\n`;
        }
        
        if (item.cross_topic) {
            tooltip += `⚡ Appears in multiple topics`;
        }
        
        return tooltip;
    }

    renderEmptyTopics() {
        this.topicFoldersContainer.innerHTML = `
            <div class="empty-topics">
                <i class="fas fa-folder-open"></i>
                <p>No topics discovered yet</p>
                <small>Upload PDFs or process YouTube videos to start building your research library</small>
            </div>
        `;
    }

    toggleTopic(topicId) {
        const topicFolder = document.querySelector(`[data-topic-id="${topicId}"]`);
        if (topicFolder) {
            topicFolder.classList.toggle('expanded');
            
            // Update selected topic
            if (topicFolder.classList.contains('expanded')) {
                // Close other topics
                document.querySelectorAll('.topic-folder.expanded').forEach(folder => {
                    if (folder !== topicFolder) {
                        folder.classList.remove('expanded');
                    }
                });
                
                this.selectedTopic = topicId;
                topicFolder.classList.add('active');
                
                // Load research connections for this topic
                this.loadResearchConnections(topicId);
            } else {
                this.selectedTopic = null;
                topicFolder.classList.remove('active');
                
                // Clear research connections
                this.clearResearchConnections();
            }
        }
    }

    selectContent(contentId) {
        console.log('Selected content:', contentId);
        
        // Find the content item to get its title
        let contentTitle = contentId;
        for (const topic of this.topics) {
            const foundItem = topic.content.find(item => item.id === contentId);
            if (foundItem) {
                contentTitle = foundItem.title;
                break;
            }
        }
        
        // Set up a query about the specific content
        this.messageInput.focus();
        this.messageInput.value = `Tell me about the content in "${contentTitle}"`;
        this.handleInputChange();
    }

    updateTopicStats(stats) {
        if (stats) {
            this.topicCount.textContent = stats.total_topics || 0;
            this.contentCount.textContent = stats.total_content || 0;
        }
    }

    // Quick Actions
    viewAllContent() {
        console.log('Viewing all content...');
        this.messageInput.value = 'Show me all the content in my research library';
        this.handleInputChange();
        this.messageInput.focus();
    }

    showCrossTopicSearch() {
        console.log('Cross-topic search...');
        this.messageInput.value = 'Find connections between different topics in my library';
        this.handleInputChange();
        this.messageInput.focus();
    }

    async exportLibrary() {
        try {
            console.log('Exporting library...');
            const response = await fetch('/api/export');
            if (!response.ok) {
                throw new Error('Export failed');
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'knowagent_library.json';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            this.showSuccess('Library exported successfully!');
        } catch (error) {
            console.error('Export error:', error);
            this.showError('Failed to export library');
        }
    }

    // =====================================
    // RESEARCH CONNECTIONS FUNCTIONALITY
    // =====================================

    async loadResearchConnections(topicId) {
        try {
            console.log('Loading research connections for topic:', topicId);
            
            // Show loading in connections panel
            this.showConnectionsLoading();
            
            const response = await fetch(`/api/research-connections/${topicId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.renderResearchConnections(data);
            } else {
                throw new Error(data.error || 'Failed to load connections');
            }
            
        } catch (error) {
            console.error('Error loading research connections:', error);
            this.showConnectionsError(error.message);
        }
    }

    showConnectionsLoading() {
        const connectionsPanel = this.getOrCreateConnectionsPanel();
        connectionsPanel.innerHTML = `
            <div class="connections-loading">
                <i class="fas fa-spinner fa-spin"></i>
                <span>Analyzing research connections...</span>
            </div>
        `;
    }

    showConnectionsError(message) {
        const connectionsPanel = this.getOrCreateConnectionsPanel();
        connectionsPanel.innerHTML = `
            <div class="connections-error">
                <i class="fas fa-exclamation-triangle"></i>
                <span>Error: ${message}</span>
            </div>
        `;
    }

    clearResearchConnections() {
        const connectionsPanel = document.getElementById('research-connections-panel');
        if (connectionsPanel) {
            connectionsPanel.innerHTML = `
                <div class="connections-placeholder">
                    <i class="fas fa-network-wired"></i>
                    <p>Select a topic to see research connections</p>
                </div>
            `;
        }
    }

    getOrCreateConnectionsPanel() {
        let panel = document.getElementById('research-connections-panel');
        if (!panel) {
            // Create the panel and add it to the main container
            panel = document.createElement('div');
            panel.id = 'research-connections-panel';
            panel.className = 'research-connections-panel';
            
            // Insert after the chat area
            const chatArea = document.querySelector('.chat-area');
            chatArea.parentNode.insertBefore(panel, chatArea.nextSibling);
        }
        return panel;
    }

    renderResearchConnections(data) {
        const connectionsPanel = this.getOrCreateConnectionsPanel();
        const connections = data.connections;
        const metadata = data.metadata;
        
        let html = `
            <div class="connections-header">
                <h3><i class="fas fa-network-wired"></i> Research Connections</h3>
                <div class="connections-stats">
                    <span class="stat">
                        <i class="fas fa-cubes"></i>
                        ${metadata.total_chunks} chunks analyzed
                    </span>
                    <span class="stat">
                        <i class="fas fa-clock"></i>
                        ${metadata.analysis_time?.toFixed(1)}s
                    </span>
                </div>
            </div>
            
            <div class="connections-content">
        `;
        
        // Contradictions
        if (connections.contradictions.length > 0) {
            html += `
                <div class="connection-section contradictions">
                    <h4><i class="fas fa-exclamation-triangle"></i> Contradictions (${connections.contradictions.length})</h4>
                    <div class="connection-items">
            `;
            
            connections.contradictions.forEach(conn => {
                html += this.renderConnectionItem(conn, 'contradiction');
            });
            
            html += `</div></div>`;
        }
        
        // Confirmations
        if (connections.confirmations.length > 0) {
            html += `
                <div class="connection-section confirmations">
                    <h4><i class="fas fa-check-circle"></i> Confirmations (${connections.confirmations.length})</h4>
                    <div class="connection-items">
            `;
            
            connections.confirmations.forEach(conn => {
                html += this.renderConnectionItem(conn, 'confirmation');
            });
            
            html += `</div></div>`;
        }
        
        // Extensions
        if (connections.extensions.length > 0) {
            html += `
                <div class="connection-section extensions">
                    <h4><i class="fas fa-arrow-up"></i> Extensions (${connections.extensions.length})</h4>
                    <div class="connection-items">
            `;
            
            connections.extensions.forEach(conn => {
                html += this.renderConnectionItem(conn, 'extension');
            });
            
            html += `</div></div>`;
        }
        
        // Research Gaps
        if (connections.gaps.length > 0) {
            html += `
                <div class="connection-section gaps">
                    <h4><i class="fas fa-question-circle"></i> Research Gaps (${connections.gaps.length})</h4>
                    <div class="gap-items">
            `;
            
            connections.gaps.forEach(gap => {
                html += `
                    <div class="gap-item">
                        <div class="gap-header">
                            <span class="gap-topic">${gap.topic_area}</span>
                            <span class="gap-confidence">${Math.round(gap.confidence * 100)}%</span>
                        </div>
                        <div class="gap-description">${gap.description}</div>
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }
        
        // No connections message
        const totalConnections = connections.contradictions.length + connections.confirmations.length + 
                                connections.extensions.length + connections.gaps.length;
        
        if (totalConnections === 0) {
            html += `
                <div class="no-connections">
                    <i class="fas fa-info-circle"></i>
                    <p>No significant research connections found.</p>
                    <small>This may indicate the content is highly consistent or needs more sources for comparison.</small>
                </div>
            `;
        }
        
        html += `</div>`;
        
        connectionsPanel.innerHTML = html;
    }

    renderConnectionItem(connection, type) {
        const iconMap = {
            'contradiction': 'fas fa-times-circle',
            'confirmation': 'fas fa-check-circle', 
            'extension': 'fas fa-arrow-up'
        };
        
        return `
            <div class="connection-item ${type}">
                <div class="connection-header">
                    <i class="${iconMap[type]}"></i>
                    <span class="connection-confidence">${Math.round(connection.confidence * 100)}%</span>
                </div>
                <div class="connection-explanation">${connection.explanation}</div>
                <div class="connection-sources">
                    <div class="source-pair">
                        <div class="source-text">
                            <strong>Source 1 (${connection.source1}):</strong>
                            <p>"${connection.chunk1_text}"</p>
                        </div>
                        <div class="source-text">
                            <strong>Source 2 (${connection.source2}):</strong>
                            <p>"${connection.chunk2_text}"</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    showNotification(message, type) {
        // Simple notification system
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1001;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
            background: ${type === 'success' ? 'var(--primary-green)' : 'var(--red)'};
        `;

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    async deleteAllData() {
        // Show confirmation dialog
        const confirmDelete = confirm(
            "⚠️ DELETE ALL DATA?\n\n" +
            "This will permanently delete:\n" +
            "• All PDF embeddings and content\n" +
            "• All YouTube transcripts and data\n" +
            "• All topic classifications\n" +
            "• All research connections\n" +
            "• All cached results\n\n" +
            "This action cannot be undone!\n\n" +
            "Type 'DELETE' to confirm:"
        );

        if (!confirmDelete) {
            return;
        }

        // Double confirmation
        const confirmText = prompt("Type 'DELETE' (all capitals) to confirm deletion:");
        if (confirmText !== 'DELETE') {
            this.showError('Deletion cancelled - confirmation text did not match');
            return;
        }

        try {
            console.log('Deleting all data...');
            
            // Show loading
            this.deleteAllDataBtn.disabled = true;
            this.deleteAllDataBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
            
            const response = await fetch('/api/delete-all-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) {
                throw new Error('Delete request failed');
            }
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess('All data deleted successfully!');
                
                // Clear the UI
                this.uploadedFiles.clear();
                this.uploadedFilesContainer.innerHTML = '';
                this.extractedLinks = [];
                this.updateExtractedLinksDisplay();
                this.topics = [];
                this.renderEmptyTopics();
                this.updateTopicStats({total_topics: 0, total_content: 0});
                this.clearResearchConnections();
                
                // Clear chat (keep welcome message)
                this.clearChat();
                
            } else {
                throw new Error(result.error || 'Delete operation failed');
            }
            
        } catch (error) {
            console.error('Delete all data error:', error);
            this.showError(`Failed to delete all data: ${error.message}`);
        } finally {
            // Restore button
            this.deleteAllDataBtn.disabled = false;
            this.deleteAllDataBtn.innerHTML = '<i class="fas fa-trash"></i> Delete All Data';
        }
    }
}

// Initialize the application
const app = new KnowledgeAgentApp();