    /**
     * Enhanced Feedback Thumbs Integration Module
     * Integrates Font Awesome thumbs up/down feedback with horizontal layout,
     * reason checkboxes for negative feedback, comment box, and submit button.
     */
    (function() {
        'use strict';
        
        // Configuration
        const FEEDBACK_CONFIG = {
            enabled: true,
            submitEndpoint: '/api/feedback',
            negativeReasons: [
                "Incorrect",
                "Incomplete",
                "Data Source Quality",
                "Irrelevant",
                "Hard to Understand",
                "Other Issue"
            ]
        };
        
        // State management
        let messageCounter = 0;
        let feedbackSubmissions = new Set(); // Track submitted feedback by message ID
        
        function generateMessageId() {
            return `msg_${Date.now()}_${++messageCounter}`;
        }
        
        function createFeedbackHTML(messageId) {
            // build checkbox list for negative reasons
            const reasonsHTML = FEEDBACK_CONFIG.negativeReasons.map(reason =>
                `<label style="display:block; margin-bottom:4px;"><input type="checkbox" class="feedback-reason" value="${reason}"> ${reason}</label>`
            ).join('');
            
            return `
                <div class="feedback-container flex flex-wrap" data-message-id="${messageId}"
                    style="display: flex; flex-direction: column; align-items: flex-end; margin-left: 8px; font-size:14px;">
                    <div class="feedback-icons inline-flex mb-5 pt-1">
                        <i class="fas fa-thumbs-up feedback-thumb" data-type="up"
                        style="color: #6b7280; cursor: pointer; margin-right: 8px; font-size: 16px; transition: color 0.2s;"
                        title="Helpful"></i>
                        <i class="fas fa-thumbs-down feedback-thumb" data-type="down"
                        style="color: #6b7280; cursor: pointer; font-size: 16px; transition: color 0.2s;"
                        title="Not helpful"></i>
                    </div>
                    <div class="feedback-details" style="display: none; margin-top: 5px; text-align: left; width: 250px;">
                        <fieldset class="dark:text-white/70" style="border:1px solid #ddd; padding:8px; border-radius:4px; margin-bottom:8px;">
                            <legend style="font-size:12px; margin-bottom:4px;">Select issues:</legend>
                            <div class="reasons-container">${reasonsHTML}</div>
                        </fieldset>
                        <div class="comment-container" style="display:none; margin-bottom:8px;">
                            <textarea class="feedback-comment text-gray-800 dark:bg-black dark:text-white"
                                    placeholder="Additional comments..."
                                    style="width:100%; box-sizing: border-box; height:60px; padding:4px; font-size:12px;border:1px solid #ddd;"></textarea>
                        </div>
                    </div>
                    <button class="feedback-submit-btn"
                            style="display: none; background: #3b82f6; color: white; border: none;
                                padding: 4px 8px; border-radius: 4px; font-size: 12px; cursor: pointer; margin-top: 5px;">
                        Submit
                    </button>
                </div>
            `;
        }
        
        function handleThumbsUp(messageId, _) {
            if (feedbackSubmissions.has(messageId)) return;
            
            const container = document.querySelector(`[data-message-id="${messageId}"]`);
            
            // Reset any previous selection
            resetFeedbackState(container);
            
            // Set thumbs up as selected
            container.querySelector('[data-type="up"]').style.color = '#22c55e';
            container.querySelector('[data-type="down"]').style.color = '#6b7280';
            container.querySelector('.feedback-details').style.display = 'none';
            container.querySelector('.feedback-submit-btn').style.display = 'inline-block';
            container.dataset.selectedType = 'positive';
        }
        
        function handleThumbsDown(messageId, _) {
            if (feedbackSubmissions.has(messageId)) return;
            
            const container = document.querySelector(`[data-message-id="${messageId}"]`);
            
            // Reset any previous selection
            resetFeedbackState(container);
            
            // Set thumbs down as selected
            container.querySelector('[data-type="down"]').style.color = '#ef4444';
            container.querySelector('[data-type="up"]').style.color = '#6b7280';
            container.querySelector('.feedback-details').style.display = 'block';
            // Submit button and comment box initially hidden, controlled by checkbox state
            container.querySelector('.comment-container').style.display = 'none'; 
            container.querySelector('.feedback-submit-btn').style.display = 'none';
            container.dataset.selectedType = 'negative';
        }
        
        function resetFeedbackState(container) {
            // Reset all checkboxes
            container.querySelectorAll('.feedback-reason').forEach(cb => {
                cb.checked = false;
            });
            
            // Clear comment
            const commentBox = container.querySelector('.feedback-comment');
            if (commentBox) {
                commentBox.value = '';
            }
            
            // Hide details and submit button
            container.querySelector('.feedback-details').style.display = 'none';
            container.querySelector('.comment-container').style.display = 'none';
            container.querySelector('.feedback-submit-btn').style.display = 'none';
            
            // Reset selected type
            delete container.dataset.selectedType;
        }
        
        function handleSubmit(messageId) {
            if (feedbackSubmissions.has(messageId)) return;
            
            const container = document.querySelector(`[data-message-id="${messageId}"]`);
            const type = container.dataset.selectedType;
            if (!type) return;
            
            // gather tags and comment
            let tags = [];
            let comment = '';
            if (type === 'positive') {
                tags = ['helpful'];
            } else {
                const checked = [...container.querySelectorAll('.feedback-reason')]
                    .filter(cb => cb.checked)
                    .map(cb => cb.value);
                if (checked.length === 0) {
                    alert('Please select at least one issue.');
                    return;
                }
                tags = checked;
                comment = container.querySelector('.feedback-comment').value.trim();
            }
            
            const feedbackData = {
                message_id: messageId,
                feedback_type: type,
                feedback_tags: tags,
                comment: comment,
                timestamp: new Date().toISOString()
            };
            
            submitFeedback(feedbackData, messageId);
        }
        
        function submitFeedback(feedbackData, messageId) {
            feedbackSubmissions.add(messageId);
            
            const botResponse = getMessageText(messageId);
            const userQuery = getUserQuery();
            const citations = getCitations(messageId);
            
            feedbackData.response = botResponse;
            feedbackData.question = userQuery;
            feedbackData.citations = citations;
            
            fetch(FEEDBACK_CONFIG.submitEndpoint, {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify(feedbackData)
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    showFeedbackConfirmation(messageId, 'Thank you for your feedback.');
                } else {
                    showFeedbackConfirmation(messageId, 'Error submitting feedback. Try again.');
                    feedbackSubmissions.delete(messageId);
                }
            })
            .catch(() => {
                showFeedbackConfirmation(messageId, 'Error submitting feedback. Try again.');
                feedbackSubmissions.delete(messageId);
            });
        }
        
        function showFeedbackConfirmation(messageId, msg) {
            const container = document.querySelector(`[data-message-id="${messageId}"]`);
            container.innerHTML = `<span style="color:#626362; font-size:12px;font-weight:bold;">${msg}</span>`;
        }
        
        function getCitations(messageId) {
            const container = document.querySelector(`[data-message-id="${messageId}"]`);
            const msgEl = container.closest('.bot-message');
            const citationEl = msgEl.querySelector('.citations-container');
            if (!citationEl) return [];

            const citations = Array.from(citationEl.querySelectorAll('.citation-item')).map(item => {
                const title = item.querySelector('.citation-title').textContent.trim();
                const url = item.querySelector('a').href;
                return { title, url };
            });
            return citations;
        }
        
        function getMessageText(messageId) {
            const container = document.querySelector(`[data-message-id="${messageId}"]`);
            const msgEl = container.closest('.bot-message');
            const txtEl = msgEl.querySelector('.message-bubble, .bot-bubble');
            return txtEl ? txtEl.textContent.trim() : '';
        }
        
        function getUserQuery() {
            // Try to find the most recent user message before the current bot message
            const messages = document.querySelectorAll('.user-message');
            if (messages.length === 0) return '';
            
            // Get the last user message
            const lastUserMsg = messages[messages.length - 1];
            const txtEl = lastUserMsg.querySelector('.message-bubble, .user-bubble');
            return txtEl ? txtEl.textContent.trim() : '';
        }
        
        function enhancedAddBotMessage(originalFn) {
            return function(message) {
                const result = originalFn.call(this, message);
                setTimeout(addFeedbackToLastMessage, 100);
                return result;
            };
        }
        
        function addFeedbackToLastMessage() {
            const bots = document.querySelectorAll('.bot-message');
            const last = bots[bots.length-1];
            if (!last || last.querySelector('.feedback-container')) return;
            const text = last.textContent.toLowerCase();
            const skip = ['developer evaluation mode enabled','standard chat mode enabled',
                        'processing evaluation','running developer evaluation'];
            if (skip.some(s => text.includes(s))) return;
            
            // Look for streaming messages that don't have the "Was this helpful?" span
            // but should get feedback buttons
            const isStreamingMessage = last.classList.contains('streaming-message') || 
                                      last.querySelector('.streaming-content');
            
            if (isStreamingMessage) {
                // Add feedback directly to streaming messages
                addFeedbackToStreamingMessage(last);
                return;
            }
            
            // Original logic for non-streaming messages
            const span = [...last.querySelectorAll('span')].find(s => s.textContent.includes('Was this helpful?'));
            if (!span) return;
            
            const msgId = generateMessageId();
            span.insertAdjacentHTML('afterend', createFeedbackHTML(msgId));
            setupListeners(msgId, last);
        }
        
        function addFeedbackToStreamingMessage(messageElement) {
            const msgId = generateMessageId();
            
            // Find the appropriate container to add feedback
            const messageContainer = messageElement.querySelector('.flex.flex-col');
            if (messageContainer) {
                // Look for existing "Was this helpful?" span
                let helpfulSpan = [...messageContainer.querySelectorAll('span')].find(s => 
                    s.textContent.includes('Was this helpful?'));
                
                // If no span exists, create one
                if (!helpfulSpan) {
                    helpfulSpan = document.createElement('span');
                    helpfulSpan.className = 'text-xs font-normal text-gray-500 dark:text-white/60 text-right pt-33';
                    helpfulSpan.textContent = 'Was this helpful?';
                    messageContainer.appendChild(helpfulSpan);
                }
                
                // Add feedback HTML
                helpfulSpan.insertAdjacentHTML('afterend', createFeedbackHTML(msgId));
                setupListeners(msgId, messageElement);
            }
        }
        
        function setupListeners(messageId, parent) {
            const container = parent.querySelector(`[data-message-id="${messageId}"]`);
            container.querySelector('[data-type="up"]').addEventListener('click', () => {
                handleThumbsUp(messageId, getMessageText(messageId));
            });
            container.querySelector('[data-type="down"]').addEventListener('click', () => {
                handleThumbsDown(messageId, getMessageText(messageId));
                // show/hide comment and submit button based on reason selection
                container.querySelectorAll('.feedback-reason').forEach(cb => {
                    cb.addEventListener('change', () => {
                        const anyChecked = [...container.querySelectorAll('.feedback-reason')].some(c => c.checked);
                        container.querySelector('.comment-container').style.display = anyChecked ? 'block' : 'none';
                        container.querySelector('.feedback-submit-btn').style.display = anyChecked ? 'inline-block' : 'none';
                    });
                });
            });
            container.querySelector('.feedback-submit-btn').addEventListener('click', () => {
                handleSubmit(messageId);
            });
            // hover
            ['up','down'].forEach(type => {
                const el = container.querySelector(`[data-type="${type}"]`);
                const hoverColor = type==='up' ? '#22c55e' : '#ef4444';
                const selectedColor = type==='up' ? '#22c55e' : '#ef4444';
                
                el.addEventListener('mouseenter', () => {
                    if (container.dataset.selectedType !== type) {
                        el.style.color = hoverColor;
                    }
                });
                el.addEventListener('mouseleave', () => {
                    if (container.dataset.selectedType === type) {
                        // Keep selected color if this is the selected type
                        el.style.color = selectedColor;
                    } else {
                        // Return to gray if not selected
                        el.style.color = '#6b7280';
                    }
                });
            });
        }
        
        function initializeFeedbackSystem() {
            if (!FEEDBACK_CONFIG.enabled) return;
            if (window.addBotMessage && typeof window.addBotMessage==='function') {
                window.addBotMessage = enhancedAddBotMessage(window.addBotMessage);
                console.log('Feedback integration initialized');
            } else {
                console.warn('addBotMessage not found; feedback disabled');
            }
        }
        
        if (document.readyState==='loading') {
            document.addEventListener('DOMContentLoaded', initializeFeedbackSystem);
        } else {
            initializeFeedbackSystem();
        }
        
        // Make key functions globally accessible
        window.addFeedbackToLastMessage = addFeedbackToLastMessage;
        
        window.FeedbackSystem = {
            config: FEEDBACK_CONFIG,
            feedbackSubmissions,
            addFeedbackToLastMessage: addFeedbackToLastMessage,
            addFeedbackToStreamingMessage: addFeedbackToStreamingMessage
        };
    })();
