import os
import pandas as pd
import csv

def create_test_data(output_path="data/test_data.csv"):
    """Create a properly formatted test dataset for the stress detection model."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create test data examples with varying stress levels
    data = [
        {"text": "I'm feeling relaxed and calm today.", "stress_level": "low"},
        {"text": "Just had a great meditation session and feel at peace.", "stress_level": "low"},
        {"text": "Work is busy but I'm handling it well.", "stress_level": "low"},
        {"text": "Taking a walk in the park helped clear my mind.", "stress_level": "low"},
        {"text": "Enjoying my coffee while watching the sunrise.", "stress_level": "low"},
        
        {"text": "I have some deadlines coming up that I'm a bit worried about.", "stress_level": "medium"},
        {"text": "Feeling somewhat anxious about my presentation tomorrow.", "stress_level": "medium"},
        {"text": "Things are piling up and it's getting harder to manage.", "stress_level": "medium"},
        {"text": "The traffic was frustrating this morning, but I'm okay.", "stress_level": "medium"},
        {"text": "My schedule is quite packed this week, but I'll manage.", "stress_level": "medium"},
        
        {"text": "I'm completely overwhelmed with everything going on.", "stress_level": "high"},
        {"text": "My anxiety is through the roof and I can't sleep.", "stress_level": "high"},
        {"text": "The stress is unbearable and I don't know how to cope.", "stress_level": "high"},
        {"text": "Everything feels like it's falling apart and I can't handle it.", "stress_level": "high"},
        {"text": "The pressure from work is making me feel like I'm breaking down.", "stress_level": "high"}
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save with proper quoting to handle any commas in the text
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    print(f"Created test dataset with {len(df)} samples")
    print(f"Saved to {output_path}")
    
    return df

if __name__ == "__main__":
    create_test_data()
    print("Test data created successfully!")