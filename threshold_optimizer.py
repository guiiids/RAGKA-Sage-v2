"""
Confidence threshold optimization for intelligent routing.
"""
import json
from typing import Dict, List, Tuple
from enhanced_pattern_matcher import EnhancedPatternMatcher
from conversation_context_analyzer import ConversationContextAnalyzer

class ThresholdOptimizer:
    """Optimize confidence thresholds based on test data."""
    
    def __init__(self):
        self.pattern_matcher = EnhancedPatternMatcher()
        self.context_analyzer = ConversationContextAnalyzer()
    
    def test_threshold_combinations(
        self,
        test_data: List[Dict],
        threshold_ranges: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Test different threshold combinations to find optimal values.
        
        Args:
            test_data: List of test cases with expected classifications
            threshold_ranges: Dict of threshold names to test values
            
        Returns:
            Best threshold combination with accuracy score
        """
        
        best_accuracy = 0.0
        best_thresholds = {}
        
        # Test all combinations
        for gpt4_threshold in threshold_ranges['gpt4_fallback']:
            for regex_threshold in threshold_ranges['regex_override']:
                
                correct_predictions = 0
                total_predictions = len(test_data)
                
                for test_case in test_data:
                    query = test_case['query']
                    expected = test_case['expected_type']
                    history = test_case.get('conversation_history', [])
                    
                    # Simulate classification with these thresholds
                    predicted = self._classify_with_thresholds(
                        query, history, gpt4_threshold, regex_threshold
                    )
                    
                    if predicted == expected:
                        correct_predictions += 1
                
                accuracy = correct_predictions / total_predictions
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_thresholds = {
                        'gpt4_fallback': gpt4_threshold,
                        'regex_override': regex_threshold,
                        'accuracy': accuracy
                    }
        
        return best_thresholds
    
    def _classify_with_thresholds(
        self,
        query: str,
        history: List[Dict],
        gpt4_threshold: float,
        regex_threshold: float
    ) -> str:
        """Simulate classification with specific thresholds."""
        
        # Get regex classification
        regex_type, regex_conf = self.pattern_matcher.classify_query(query, history)
        
        # High-confidence regex override
        if regex_conf >= regex_threshold:
            return regex_type
        
        # Simulate GPT-4 classification (would need real data for this)
        # For now, use regex with lower confidence
        if regex_conf >= gpt4_threshold:
            return regex_type
        
        # Default fallback
        return 'NEW_TOPIC_INFORMATIONAL'

# Test data for threshold optimization
TEST_CASES = [
    {
        'query': 'How do I create a calendar?',
        'expected_type': 'NEW_TOPIC_PROCEDURAL',
        'conversation_history': []
    },
    {
        'query': 'What is a calendar?',
        'expected_type': 'NEW_TOPIC_INFORMATIONAL',
        'conversation_history': []
    },
    {
        'query': 'Tell me more about that',
        'expected_type': 'CONTEXTUAL_FOLLOW_UP',
        'conversation_history': [
            {'role': 'user', 'content': 'What is a calendar?'},
            {'role': 'assistant', 'content': 'A calendar is...'}
        ]
    },
    {
        'query': 'What was my first question?',
        'expected_type': 'HISTORY_RECALL',
        'conversation_history': [
            {'role': 'user', 'content': 'How do I create a calendar?'},
            {'role': 'assistant', 'content': 'To create a calendar...'}
        ]
    },
    {
        'query': 'then what?',
        'expected_type': 'CONTEXTUAL_FOLLOW_UP',
        'conversation_history': [
            {'role': 'user', 'content': 'How do I create a calendar?'},
            {'role': 'assistant', 'content': 'First, go to settings...'}
        ]
    },
    {
        'query': 'what about the first one?',
        'expected_type': 'CONTEXTUAL_FOLLOW_UP',
        'conversation_history': [
            {'role': 'user', 'content': 'What are the calendar options?'},
            {'role': 'assistant', 'content': 'There are three options...'}
        ]
    },
    {
        'query': 'can you tell me more?',
        'expected_type': 'CONTEXTUAL_FOLLOW_UP',
        'conversation_history': [
            {'role': 'user', 'content': 'What is calendar sharing?'},
            {'role': 'assistant', 'content': 'Calendar sharing allows...'}
        ]
    },
    {
        'query': 'what did we discuss earlier?',
        'expected_type': 'HISTORY_RECALL',
        'conversation_history': [
            {'role': 'user', 'content': 'How do I set permissions?'},
            {'role': 'assistant', 'content': 'To set permissions...'},
            {'role': 'user', 'content': 'What about notifications?'},
            {'role': 'assistant', 'content': 'For notifications...'}
        ]
    },
    {
        'query': 'go back to the previous topic',
        'expected_type': 'HISTORY_RECALL',
        'conversation_history': [
            {'role': 'user', 'content': 'How do I create events?'},
            {'role': 'assistant', 'content': 'To create events...'},
            {'role': 'user', 'content': 'What about recurring events?'},
            {'role': 'assistant', 'content': 'For recurring events...'}
        ]
    },
    {
        'query': 'how to share a calendar',
        'expected_type': 'NEW_TOPIC_PROCEDURAL',
        'conversation_history': []
    },
    {
        'query': 'explain calendar permissions',
        'expected_type': 'NEW_TOPIC_INFORMATIONAL',
        'conversation_history': []
    },
    {
        'query': 'steps to add an event',
        'expected_type': 'NEW_TOPIC_PROCEDURAL',
        'conversation_history': []
    },
    {
        'query': 'why',
        'expected_type': 'CONTEXTUAL_FOLLOW_UP',
        'conversation_history': [
            {'role': 'user', 'content': 'Should I use shared calendars?'},
            {'role': 'assistant', 'content': 'It depends on your needs...'}
        ]
    },
    {
        'query': 'how',
        'expected_type': 'CONTEXTUAL_FOLLOW_UP',
        'conversation_history': [
            {'role': 'user', 'content': 'Can I sync calendars?'},
            {'role': 'assistant', 'content': 'Yes, you can sync calendars...'}
        ]
    },
    {
        'query': 'any other examples?',
        'expected_type': 'CONTEXTUAL_FOLLOW_UP',
        'conversation_history': [
            {'role': 'user', 'content': 'What are some calendar use cases?'},
            {'role': 'assistant', 'content': 'Here are some examples...'}
        ]
    }
]

if __name__ == "__main__":
    optimizer = ThresholdOptimizer()
    
    threshold_ranges = {
        'gpt4_fallback': [0.3, 0.4, 0.5, 0.6, 0.7],
        'regex_override': [0.7, 0.75, 0.8, 0.85, 0.9]
    }
    
    print("Testing threshold combinations...")
    best_thresholds = optimizer.test_threshold_combinations(TEST_CASES, threshold_ranges)
    
    print("\n" + "="*50)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best thresholds: {json.dumps(best_thresholds, indent=2)}")
    
    # Test individual queries with best thresholds
    print(f"\nTesting individual queries with optimal thresholds:")
    print(f"GPT-4 fallback threshold: {best_thresholds['gpt4_fallback']}")
    print(f"Regex override threshold: {best_thresholds['regex_override']}")
    print(f"Overall accuracy: {best_thresholds['accuracy']:.2%}")
    
    print(f"\nSample classifications:")
    for i, test_case in enumerate(TEST_CASES[:5]):
        query = test_case['query']
        expected = test_case['expected_type']
        history = test_case.get('conversation_history', [])
        
        predicted = optimizer._classify_with_thresholds(
            query, history, 
            best_thresholds['gpt4_fallback'], 
            best_thresholds['regex_override']
        )
        
        status = "✓" if predicted == expected else "✗"
        print(f"{status} '{query}' -> {predicted} (expected: {expected})")
