#!/usr/bin/env python3
import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    import_checks = {
        'flask': 'flask',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',  # sklearn is the import name
        'joblib': 'joblib'
    }
    
    missing = []
    for import_name, package_name in import_checks.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

def setup_directories():
    """Create required directories"""
    dirs = ['models', 'logs', 'uploads', 'static/charts']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Directories created")

def run_app():
    """Run the Flask application"""
    print("🚀 Starting FraudShield application...")
    print("📱 Open your browser and go to: http://localhost:5001")
    
    # Import and run the app
    from app import app
    app.run(debug=True, port=5001, host='0.0.0.0')

if __name__ == '__main__':
    print("🔍 FraudShield - E-commerce Fraud Detection System")
    print("=" * 50)
    
    if not check_requirements():
        sys.exit(1)
    
    setup_directories()
    run_app()

