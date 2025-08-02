/**
 * Streaming Chat Implementation
 * Replaces the standard fetch-based chat with streaming responses
 */

/**
 * Robust citation-safe formatter injected for streaming chat.
 * Guarantees all [n] citations map to current message sources only.
 */
window.formatMessage = function formatMessage(text) {
  // Determine sources for the current message
  const sources =
    (window.currentMessageId &&
      window.messageSourcesMap &&
      window.messageSourcesMap[window.currentMessageId]) ||
    window.lastSources;

  // Fallback: If no sources, return as-is, process markdown quickly
  if (!sources || sources.length === 0) {
    return text.replace(/\n/g, '<br>');
  }

  // [n] style marker rewritten to <sup><a ...> if possible, otherwise left alone
  // Map of display_id to source in current message
  const sourceMap = {};
  sources.forEach(src => {
    sourceMap[String(src.display_id)] = src;
  });

  // Replace [n] with clickable anchor, message-scoped
  const replaced = text.replace(/\[(\d+)\]/g, (match, p1) => {
    const source = sourceMap[p1];
    if (!source) return match; // don't break unknown ones
    // The anchor uses the message-scoped citationId for exact handler match
    const citationId = source.scopedId || (window.currentMessageId + '-' + p1);
    return `<sup><a href="javascript:void(0);" onclick="handleMessageScopedCitationClick('${citationId}')" class="citation-link" data-citation-id="${citationId}">[${p1}]</a></sup>`;
  });
  return replaced.replace(/\n/g, '<br>');
};
// Global flag to enable/disable streaming
window.streamingEnabled = true;

// Store the current streaming message element for real-time updates
let currentStreamingMessage = null;

// Message-scoped citation system
window.messageSourcesMap = {}; // message_id -> sources array
// Backwards compatibility alias
window.messageSources = window.messageSourcesMap;
window.currentMessageId = null;
window.messageCounter = 0;

// Debug flag for citation system
window.debugCitations = true;

/**
 * Enhanced submitQuery function with streaming support
 */
function submitQueryWithStreaming() {
  const query = queryInput.value.trim();
  if (!query) return;
  
  // Check if query is enhanced (from magic wand)
  const isEnhanced = queryInput.dataset.enhanced === 'true';
  
  addUserMessage(query);
  queryInput.value = '';
  queryInput.dataset.enhanced = 'false'; // Reset enhanced flag
  
  // Reset textarea height to 1 line after sending
  const lineHeight = parseInt(window.getComputedStyle(queryInput).lineHeight);
  queryInput.style.height = 'auto';
  queryInput.style.height = lineHeight + 'px';
  
  // Disable submit button and show spinner
  const submitBtn = document.getElementById('submit-btn');
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
  }
  
  // Show typing indicator initially
  const typingIndicator = addTypingIndicator();
  
  // Start streaming
  streamRAGResponse(query, isEnhanced, typingIndicator);
}

/**
 * Stream RAG response using fetch with ReadableStream
 */
async function streamRAGResponse(query, isEnhanced = false, typingIndicator = null) {
  try {
    // Get current settings for the request
    const settings = getCurrentSettings();
    
    const response = await fetch('/api/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query: query,
        is_enhanced: isEnhanced,
        settings: settings
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Remove typing indicator and create streaming message container
    if (typingIndicator) typingIndicator.remove();
    currentStreamingMessage = createStreamingMessageContainer();
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentContent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Decode the chunk and add to buffer
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete lines from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'content') {
              // Append new content and update the message
              currentContent += data.data;
              updateStreamingMessage(currentContent);
            } else if (data.type === 'metadata') {
              // Handle sources and evaluation
              handleStreamingMetadata(data.data);
            } else if (data.type === 'done') {
              // Streaming complete
              finalizeStreamingMessage();
              return;
            } else if (data.type === 'error') {
              throw new Error(data.data);
            }
          } catch (parseError) {
            console.warn('Failed to parse streaming data:', line, parseError);
          }
        }
      }
    }
    
    // Finalize message if we exit the loop
    finalizeStreamingMessage();
    
  } catch (error) {
    console.error('Streaming error:', error);
    
    // Remove typing indicator if still present
    if (typingIndicator) typingIndicator.remove();
    
    // Remove streaming message if it exists
    if (currentStreamingMessage) {
      currentStreamingMessage.remove();
      currentStreamingMessage = null;
    }
    
    // Show error message
    addBotMessage('Error: Could not connect to server. Please try again later.');
    
    // Restore submit button
    restoreSubmitButton();
  }
}

/**
 * Create a streaming message container that can be updated in real-time
 */
function createStreamingMessageContainer() {
  // Hide center logo if visible
  const centerLogo = document.getElementById('center-logo');
  if (centerLogo && !centerLogo.classList.contains('hidden')) {
    centerLogo.classList.add('hidden');
  }
  
  const messageDiv = document.createElement('div');
  messageDiv.className = 'bot-message streaming-message';
  messageDiv.innerHTML = `
    <img class="w-8 h-8 rounded-full" src="https://content.tst-34.aws.agilent.com/wp-content/uploads/2025/05/dalle.png" alt="AI Agent">
    <div class="flex flex-col w-auto max-w-[90%] leading-1.5">
      <div class="flex items-center space-x-2 rtl:space-x-reverse pl-1 pb-1">
        <span class="text-xs font-semibold text-gray-900 dark:text-white">SAGE<span class="mt-1 text-xs leading-tight font-strong text-blue-700 dark:text-white/80"> AI Agent</span></span>
        <span class="streaming-indicator text-xs text-blue-500">●</span>
      </div>
      <div class="streaming-content text-sm leading-6 font-normal py-2 text-gray-900 dark:text-white/80 space-y-4 message-bubble bot-bubble">
        <div class="typing-cursor">|</div>
      </div>
    </div>
  `;
  
  // Add CSS for streaming indicator and cursor
  if (!document.getElementById('streaming-styles')) {
    const style = document.createElement('style');
    style.id = 'streaming-styles';
    style.textContent = `
      .streaming-indicator {
        animation: pulse 1s infinite;
      }
      .typing-cursor {
        animation: blink 1s infinite;
        display: inline;
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
      }
      @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
      }
    `;
    document.head.appendChild(style);
  }
  
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  
  return messageDiv;
}

/**
 * Update the streaming message with new content
 */
function updateStreamingMessage(content) {
  if (!currentStreamingMessage) return;
  
  const contentDiv = currentStreamingMessage.querySelector('.streaming-content');
  if (contentDiv) {
    // Combine citation processing with full Markdown parsing, then add cursor
    // Parse raw markdown first for full HTML rendering and then insert citations
    let html = (typeof marked !== 'undefined' ? marked.parse(content) : content);
    // Inject citation links into HTML after markdown parsing
    const currentSources =
      (window.currentMessageId &&
        window.messageSourcesMap &&
        window.messageSourcesMap[window.currentMessageId]) ||
      [];
    html = html.replace(/\[(\d+)\]/g, (match, p1) => {
      const source = currentSources.find(src => String(src.display_id) === p1);
      if (!source) return match;
      const citationId = source.scopedId || `${window.currentMessageId}-${p1}`;
      return `<sup><a href="javascript:void(0);" onclick="handleMessageScopedCitationClick('${citationId}')" class="citation-link" data-citation-id="${citationId}">[${p1}]</a></sup>`;
    });
    contentDiv.innerHTML = html + '<span class="typing-cursor">|</span>';
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

/**
 * Handle metadata from streaming (sources, evaluation, etc.)
 */
function handleStreamingMetadata(metadata) {
  if (metadata.sources && metadata.sources.length > 0) {
    // Increment message counter for new messages
    window.messageCounter++;
    window.currentMessageId = window.messageCounter;
    
    // Store sources with proper citation IDs from PostgreSQL
    const messageScopedSources = metadata.sources.map(source => ({
      ...source,
      // Use the citation_id from PostgreSQL service
      messageId: window.currentMessageId,
      scopedId: `${window.currentMessageId}-${source.citation_id || source.display_id || '1'}`,
      // Keep the original citation_id from PostgreSQL
      citation_id: source.citation_id || source.display_id || '1',
      display_id: source.display_id || source.citation_id || '1'
    }));
    
    // Store sources for this specific message
    window.messageSourcesMap[window.currentMessageId] = messageScopedSources;

    // Also update lastSources for backward compatibility
    window.lastSources = messageScopedSources;
    
    // DEBUG: Enhanced console logging for debugging
    if (window.debugCitations) {
      console.group(`🔍 CITATION DEBUG - Message ${window.currentMessageId}`);
      console.log('📥 Raw metadata sources:', metadata.sources);
      console.log('🏷️ Message-scoped sources:', messageScopedSources.map(s => ({
        messageId: s.messageId,
        scopedId: s.scopedId,
        uniqueId: s.id,
        displayId: s.display_id,
        title: s.title.substring(0, 30) + '...'
      })));
      console.log('📚 All message sources:', Object.keys(window.messageSourcesMap).map(msgId => ({
        messageId: msgId,
        sourceCount: window.messageSourcesMap[msgId].length,
        sources: window.messageSourcesMap[msgId].map(s => s.scopedId)
      })));
      console.log('🔗 Source mapping for frontend:', {
        currentMessageId: window.currentMessageId,
        totalMessages: Object.keys(window.messageSourcesMap).length,
        totalSources: Object.values(window.messageSourcesMap).reduce((sum, sources) => sum + sources.length, 0)
      });
      console.groupEnd();
    }
  }
  
  // Handle other metadata like evaluation if needed
  if (metadata.evaluation) {
    console.log('📊 Evaluation received:', metadata.evaluation);
  }
}

/**
 * Finalize the streaming message (remove cursor, add sources, etc.)
 */
function finalizeStreamingMessage() {
  if (!currentStreamingMessage) return;
  
  // Remove streaming indicator and cursor
  const indicator = currentStreamingMessage.querySelector('.streaming-indicator');
  if (indicator) indicator.remove();
  
  const contentDiv = currentStreamingMessage.querySelector('.streaming-content');
  if (contentDiv) {
    // Remove cursor from content
    contentDiv.innerHTML = contentDiv.innerHTML.replace('<span class="typing-cursor">|</span>', '');
  }
  
  // Add "Was this helpful?" span for feedback system
  const messageContainer = currentStreamingMessage.querySelector('.flex.flex-col');
  if (messageContainer) {
    const helpfulSpan = document.createElement('span');
    helpfulSpan.className = 'text-xs font-normal text-gray-500 dark:text-white/60 text-right pt-33';
    helpfulSpan.textContent = 'Was this helpful?';
    messageContainer.appendChild(helpfulSpan);
  }
  
  // Add sources if available
  const finalizedSources =
    (window.messageSourcesMap &&
      window.currentMessageId &&
      window.messageSourcesMap[window.currentMessageId]) ||
    [];
  if (finalizedSources.length > 0) {
    addSourcesUtilizedSection(finalizedSources, window.currentMessageId);
  }
  
  // Remove streaming class
  currentStreamingMessage.classList.remove('streaming-message');
  
  // Manually trigger feedback system for streaming messages
  setTimeout(() => {
    if (window.FeedbackSystem && typeof addFeedbackToLastMessage === 'function') {
      addFeedbackToLastMessage();
    }
  }, 100);
  
  currentStreamingMessage = null;
  
  // Restore submit button
  restoreSubmitButton();
}

/**
 * Restore the submit button to its original state
 */
function restoreSubmitButton() {
  const submitBtn = document.getElementById('submit-btn');
  if (submitBtn) {
    submitBtn.disabled = false;
    submitBtn.innerHTML = 'Send';
  }
}

/**
 * Get current settings from the settings form
 */
function getCurrentSettings() {
  const settings = {};
  
  // Get custom prompt if available
  const customPrompt = document.getElementById('custom-prompt');
  if (customPrompt && customPrompt.value.trim()) {
    settings.custom_prompt = customPrompt.value.trim();
  }
  
  // Get prompt mode
  const promptModeRadios = document.querySelectorAll('input[name="prompt-mode"]');
  for (const radio of promptModeRadios) {
    if (radio.checked) {
      settings.system_prompt_mode = radio.value;
      break;
    }
  }
  
  // Get developer settings
  const temperature = document.getElementById('dev-temperature');
  if (temperature) {
    settings.temperature = parseFloat(temperature.value);
  }
  
  const topP = document.getElementById('dev-top-p');
  if (topP) {
    settings.top_p = parseFloat(topP.value);
  }
  
  const maxTokens = document.getElementById('dev-max-tokens');
  if (maxTokens) {
    settings.max_completion_tokens = parseInt(maxTokens.value);
  }
  
  return settings;
}

/**
 * Enhanced citation click handler with PostgreSQL citation service support
 */
function handleMessageScopedCitationClick(sourceId) {
  // Guard against invalid source identifiers
  if (!sourceId || typeof sourceId !== 'string') {
    if (window.debugCitations) {
      console.warn('❗ Invalid sourceId in handleMessageScopedCitationClick:', sourceId);
    }
    return;
  }

  // Normalize to primitive string before regex operations
  sourceId = String(sourceId);

  if (window.debugCitations) {
    console.group(`🖱️ CITATION CLICK DEBUG`);
    console.log('📌 Clicked source ID:', sourceId);
  }

  // Try to extract message ID and citation ID from scoped citation (e.g., "1-2" -> message 1, citation 2)
  const scopedMatch = sourceId.match(/^(\d+)-(\d+)$/);
  if (scopedMatch) {
    const [, messageId, citationId] = scopedMatch;
    
    if (window.debugCitations) {
      console.log('🎯 Detected scoped citation:', { messageId, citationId });
    }
    
    // First try to find in local message sources cache
    const messageSources = (window.messageSourcesMap && window.messageSourcesMap[messageId]) || [];
    const localSource = messageSources.find(s => 
      s.scopedId === sourceId || s.citation_id == citationId || s.display_id == citationId
    );
    
    if (localSource) {
      if (window.debugCitations) {
        console.log('✅ Found source in local cache:', {
          messageId: localSource.messageId,
          title: localSource.title.substring(0, 50) + '...',
          citationId: localSource.citation_id
        });
        console.groupEnd();
      }
      
      showSourcePopup(sourceId, localSource.title, localSource.content);
      return true;
    }
    
    // If not found locally, try PostgreSQL citation service
    if (window.debugCitations) {
      console.log('🔄 Fetching from PostgreSQL citation service...');
    }
    
    fetch(`/api/session-citations/get?citation_id=${citationId}`)
      .then(response => response.json())
      .then(data => {
        if (window.debugCitations) {
          console.log('📥 PostgreSQL response:', data);
          console.groupEnd();
        }
        
        if (data.success && data.source) {
          showSourcePopup(sourceId, data.source.title, data.source.content);
        } else {
          showSourcePopup(sourceId, 'Source [null] - Not Found', 'Error: Source information not found.');
        }
      })
      .catch(error => {
        console.error('Error fetching citation from PostgreSQL:', error);
        if (window.debugCitations) {
          console.groupEnd();
        }
        showSourcePopup(sourceId, 'Source [null] - Not Found', 'Error: Source information not found.');
      });
    
    return true;
  }
  
  // Final fallback: Search all message sources
  for (const messageId in window.messageSourcesMap) {
    const sources = window.messageSourcesMap[messageId];
    const source = sources.find(s => 
      s.id === sourceId || s.display_id === sourceId || s.scopedId === sourceId || s.citation_id == sourceId
    );
    
    if (source) {
      if (window.debugCitations) {
        console.log('✅ Found source in all messages search:', {
          foundInMessage: messageId,
          title: source.title.substring(0, 50) + '...'
        });
        console.groupEnd();
      }
      
      showSourcePopup(sourceId, source.title, source.content);
      return true;
    }
  }
  
  if (window.debugCitations) {
    console.error('❌ Source not found anywhere:', sourceId);
    console.log('🔍 Available sources:', {
      messageSources: Object.keys(window.messageSourcesMap).length,
      allSources: Object.values(window.messageSourcesMap).flat().map(s => ({
        scopedId: s.scopedId, 
        citationId: s.citation_id, 
        displayId: s.display_id 
      }))
    });
    console.groupEnd();
  }
  
  // Show not found popup
  showSourcePopup(sourceId, 'Source [null] - Not Found', 'Error: Source information not found.');
  return false;
}

/**
 * Show all sources from the entire conversation
 */
function showAllConversationSources() {
  const allSources = [];
  
  for (const messageId in window.messageSourcesMap) {
    const sources = window.messageSourcesMap[messageId];
    sources.forEach(source => {
      allSources.push({
        ...source,
        messageContext: `Message ${messageId}`
      });
    });
  }
  
  if (allSources.length === 0) {
    console.log('No sources available in conversation');
    return;
  }
  
  // Log all sources for debugging
  console.group('📚 ALL CONVERSATION SOURCES');
  console.table(allSources.map(s => ({
    messageContext: s.messageContext,
    scopedId: s.scopedId,
    title: s.title.substring(0, 40) + '...',
    uniqueId: s.id
  })));
  console.groupEnd();
  
  // Could implement a modal to display all sources here
  // For now, just log them to console
}

/**
 * Debug helper: Print current citation state
 */
function debugCitationState() {
  console.group('🔧 CITATION SYSTEM DEBUG STATE');
  console.log('💬 Message Counter:', window.messageCounter);
  console.log('🎯 Current Message ID:', window.currentMessageId);
  console.log('📚 Message Sources Count:', Object.keys(window.messageSourcesMap).length);
  
  console.log('📊 Detailed Message Sources:');
  for (const [msgId, sources] of Object.entries(window.messageSourcesMap)) {
    console.log(`  Message ${msgId}:`, sources.map(s => ({
      scopedId: s.scopedId,
      displayId: s.display_id,
      title: s.title.substring(0, 30) + '...'
    })));
  }
  console.groupEnd();
}

/**
 * Override the global submitQuery function to use streaming
 */
function initializeStreamingChat() {
  // Replace the global submitQuery function
  if (typeof window.submitQuery === 'function') {
    window.originalSubmitQuery = window.submitQuery;
  }
  
  // Set the new streaming function as the global submitQuery
  window.submitQuery = submitQueryWithStreaming;
  
  // Expose enhanced citation handler globally
  window.handleCitationClick = handleMessageScopedCitationClick;
  window.showAllConversationSources = showAllConversationSources;
  window.debugCitationState = debugCitationState;
  
  console.log('Streaming chat initialized with enhanced citation system');
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  // Wait a bit to ensure other scripts have loaded
  setTimeout(initializeStreamingChat, 100);
});

// Also initialize immediately if DOM is already loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeStreamingChat);
} else {
  initializeStreamingChat();
}
