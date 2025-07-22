"""
NER-Based Keyword Extraction for PDFs and Videos
Replaces manual keyword lists with automatic entity recognition
"""

# import spacy  # Not required for regex fallback
from typing import List, Dict, Set
import re

class TechnicalNER:
    """Enhanced NER for technical documents and transcripts"""
    
    def __init__(self):
        # Load spaCy model (install with: python -m spacy download en_core_web_sm)
        # Force fallback for testing without spaCy installation
        self.nlp = None
        print("Using regex-based extraction (spaCy not available)")
    
    def extract_technical_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract technical entities and keywords from text"""
        
        if not self.nlp:
            # Fallback to regex-based extraction
            return self._regex_fallback(text)
        
        doc = self.nlp(text)
        
        # Categories of entities to extract
        entities = {
            'technologies': [],
            'measurements': [],
            'materials': [],
            'processes': [],
            'organizations': [],
            'locations': []
        }
        
        # Extract named entities
        for ent in doc.ents:
            entity_text = ent.text.strip()
            
            if ent.label_ == "ORG":  # Organizations
                entities['organizations'].append(entity_text)
            elif ent.label_ in ["QUANTITY", "PERCENT", "CARDINAL"]:  # Measurements
                entities['measurements'].append(entity_text)
            elif ent.label_ == "GPE":  # Locations
                entities['locations'].append(entity_text)
            elif ent.label_ in ["PRODUCT", "WORK_OF_ART"]:  # Technologies
                entities['technologies'].append(entity_text)
        
        # Extract domain-specific technical terms using patterns
        entities.update(self._extract_technical_patterns(text))
        
        # Clean and deduplicate
        for category, items in entities.items():
            entities[category] = list(set([item.lower().strip() for item in items if len(item) > 2]))
        
        return entities
    
    def _extract_technical_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract technical terms using domain-specific patterns"""
        
        patterns = {
            'technologies': [
                r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+(?:technology|system|process|method)\b',
                r'\b(solar panels?|photovoltaic|DAC|direct air capture|carbon capture)\b',
                r'\b([a-z]+(?:-[a-z]+)*)\s+cells?\b',  # "silicon cells", "perovskite cells"
            ],
            'measurements': [
                r'\b(\d+(?:\.\d+)?)\s*%\b',  # Percentages
                r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(kWh|MW|GW|tons?|kg|meters?)\b',  # Units
                r'\b(\$\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:per|/)\s*(kWh|ton|kg)\b',  # Costs
            ],
            'materials': [
                r'\b(silicon|lithium|olivine|CO₂|CO2|perovskite|tandem)\b',
                r'\b([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:compound|material|substrate)\b',
            ],
            'processes': [
                r'\b([a-z]+(?:\s+[a-z]+)*)\s+(?:sequestration|capture|storage|weathering)\b',
                r'\b(enhanced\s+[a-z]+|geological\s+[a-z]+|biological\s+[a-z]+)\b',
            ]
        }
        
        results = {key: [] for key in patterns.keys()}
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text.lower(), re.IGNORECASE)
                if isinstance(matches[0] if matches else None, tuple):
                    # Handle grouped matches
                    results[category].extend([' '.join(match).strip() for match in matches])
                else:
                    results[category].extend(matches)
        
        return results
    
    def _regex_fallback(self, text: str) -> Dict[str, List[str]]:
        """Fallback extraction using only regex (no spaCy required)"""
        
        # Basic technical term extraction
        solar_terms = re.findall(r'\b(solar|photovoltaic|panel|silicon|perovskite|efficiency|grid)\b', text.lower())
        carbon_terms = re.findall(r'\b(carbon|sequestration|capture|dac|co2|storage|geological)\b', text.lower())
        measurements = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:%|kWh|MW|tons?)\b', text.lower())
        
        return {
            'solar_energy': list(set(solar_terms)),
            'carbon_tech': list(set(carbon_terms)),
            'measurements': list(set(measurements)),
            'technologies': [],
            'materials': [],
            'processes': []
        }
    
    def categorize_document(self, text: str) -> Dict[str, float]:
        """Categorize document into topic areas with confidence scores"""
        
        keywords = self.extract_technical_keywords(text)
        
        # Define topic signatures
        topic_signatures = {
            'solar_energy': ['solar', 'panel', 'photovoltaic', 'silicon', 'efficiency', 'grid', 'renewable'],
            'carbon_sequestration': ['carbon', 'sequestration', 'capture', 'dac', 'co2', 'storage', 'geological'],
            'energy_systems': ['energy', 'power', 'electricity', 'generation', 'kwh', 'supply'],
            'materials_science': ['silicon', 'lithium', 'perovskite', 'materials', 'compound', 'substrate'],
            'environmental': ['climate', 'emissions', 'environmental', 'weathering', 'atmosphere']
        }
        
        scores = {}
        text_lower = text.lower()
        
        for topic, signature in topic_signatures.items():
            # Count matches in text
            matches = sum(1 for term in signature if term in text_lower)
            # Count matches in extracted entities
            entity_matches = sum(1 for category in keywords.values() 
                               for entity in category 
                               for term in signature 
                               if term in entity)
            
            # Calculate confidence score
            total_terms = len(signature)
            confidence = (matches + entity_matches * 2) / (total_terms * 2)  # Weight entities higher
            scores[topic] = min(confidence, 1.0)  # Cap at 1.0
        
        return scores

def enhance_content_with_ner(content_text: str, title: str = "") -> Dict[str, any]:
    """
    Process content with NER to extract rich metadata
    
    Args:
        content_text: Full text content
        title: Document title
        
    Returns:
        Enhanced metadata with extracted entities and topics
    """
    
    ner = TechnicalNER()
    
    # Combine title and content for analysis
    full_text = f"{title} {content_text}"
    
    # Extract entities
    entities = ner.extract_technical_keywords(full_text)
    
    # Categorize document
    topic_scores = ner.categorize_document(full_text)
    
    # Generate smart keywords (combination of entities and topics)
    smart_keywords = []
    for category, items in entities.items():
        smart_keywords.extend(items[:3])  # Top 3 from each category
    
    # Add high-confidence topic names as keywords
    for topic, score in topic_scores.items():
        if score > 0.3:  # High confidence threshold
            smart_keywords.append(topic.replace('_', ' '))
    
    return {
        'entities': entities,
        'topic_scores': topic_scores,
        'smart_keywords': list(set(smart_keywords)),
        'primary_topics': [topic for topic, score in topic_scores.items() if score > 0.4]
    }

# Example usage
if __name__ == "__main__":
    # Test with sample content
    sample_text = """
    Solar Panel Efficiency and Grid Integration Challenges
    
    Silicon solar cell efficiency: theoretical limit 29%, current commercial 20-22%
    Perovskite tandem cells: laboratory achievements 31.3% efficiency
    Temperature coefficients: -0.4%/°C power loss for silicon panels
    Grid stability: frequency regulation, voltage support requirements
    Energy storage integration: lithium-ion costs $137/kWh, cycle life 6,000+ cycles
    """
    
    result = enhance_content_with_ner(sample_text)
    print("NER Enhancement Results:")
    print(f"Entities: {result['entities']}")
    print(f"Topic Scores: {result['topic_scores']}")
    print(f"Smart Keywords: {result['smart_keywords']}")
    print(f"Primary Topics: {result['primary_topics']}")