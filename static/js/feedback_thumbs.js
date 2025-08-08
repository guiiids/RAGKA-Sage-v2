/**
 * Feedback Thumbs Utility Functions
 * Additional utilities for the feedback system
 */

// Utility functions for feedback system
window.FeedbackUtils = {
    /**
     * Get feedback statistics for the current session
     */
    getSessionStats: function() {
        const containers = document.querySelectorAll('.feedback-container');
        let positive = 0, negative = 0, total = 0;
        
        containers.forEach(container => {
            if (container.dataset.selectedType === 'positive') positive++;
            else if (container.dataset.selectedType === 'negative') negative++;
            total++;
        });
        
        return { positive, negative, total };
    },
    
    /**
     * Check if a message has feedback
     */
    hasFeedback: function(messageId) {
        const container = document.querySelector(`[data-message-id="${messageId}"]`);
        return container && container.dataset.selectedType;
    },
    
    /**
     * Reset all feedback in the current session
     */
    resetAllFeedback: function() {
        const containers = document.querySelectorAll('.feedback-container');
        containers.forEach(container => {
            // Reset colors
            const upBtn = container.querySelector('[data-type="up"]');
            const downBtn = container.querySelector('[data-type="down"]');
            if (upBtn) upBtn.style.color = '#6b7280';
            if (downBtn) downBtn.style.color = '#6b7280';
            
            // Hide details
            const details = container.querySelector('.feedback-details');
            if (details) details.style.display = 'none';
            
            // Reset checkboxes
            container.querySelectorAll('.feedback-reason').forEach(cb => {
                cb.checked = false;
            });
            
            // Clear comments
            const comment = container.querySelector('.feedback-comment');
            if (comment) comment.value = '';
            
            // Reset state
            delete container.dataset.selectedType;
        });
        
        // Clear submission tracking
        if (window.FeedbackSystem) {
            window.FeedbackSystem.feedbackSubmissions.clear();
        }
    },
    
    /**
     * Export feedback data for analysis
     */
    exportFeedbackData: function() {
        const containers = document.querySelectorAll('.feedback-container');
        const data = [];
        
        containers.forEach(container => {
            const messageId = container.dataset.messageId;
            const selectedType = container.dataset.selectedType;
            
            if (selectedType) {
                const feedbackData = {
                    messageId,
                    type: selectedType,
                    timestamp: new Date().toISOString()
                };
                
                if (selectedType === 'negative') {
                    const reasons = [...container.querySelectorAll('.feedback-reason')]
                        .filter(cb => cb.checked)
                        .map(cb => cb.value);
                    const comment = container.querySelector('.feedback-comment')?.value || '';
                    
                    feedbackData.reasons = reasons;
                    feedbackData.comment = comment;
                }
                
                data.push(feedbackData);
            }
        });
        
        return data;
    }
};

// Console commands for debugging
if (typeof window !== 'undefined' && window.console) {
    window.console.log('Feedback utilities loaded. Use FeedbackUtils.getSessionStats() to see feedback statistics.');
}
