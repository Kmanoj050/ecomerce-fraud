from flask import Flask, render_template, request, redirect, send_file, jsonify, session
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import joblib
from datetime import datetime
import json
import hashlib
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change for production

# Configuration
UPLOAD_FOLDER = 'uploads'
LOG_PATH = 'logs/prediction_log.csv'
MODEL_DIR = 'models'
PERFORMANCE_PATH = os.path.join(MODEL_DIR, 'performance_metrics.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")

# Load model metadata if available
try:
    with open(os.path.join(MODEL_DIR, 'model_metadata.json')) as f:
        MODEL_METADATA = json.load(f)
except:
    MODEL_METADATA = {
        'features': {
            'categorical': ['name', 'item', 'city', 'payment_mode'],
            'numeric': ['price', 'quantity']
        }
    }

# Helper functions
def generate_temporal_features():
    """Generate temporal features based on current time"""
    now = datetime.now()
    return {
        'hour': now.hour,
        'day_of_week': now.strftime('%A'),
        'month': now.strftime('%B')
    }

def generate_feature_hash(features):
    """Generate unique hash for a set of features"""
    feature_str = ''.join(str(v) for v in features.values())
    return hashlib.md5(feature_str.encode()).hexdigest()

def load_performance_metrics():
    """Load model performance metrics"""
    if os.path.exists(PERFORMANCE_PATH):
        with open(PERFORMANCE_PATH) as f:
            return json.load(f)
    return {}

# Routes
@app.route('/')
def home():
    # Get recent history data
    recent_records = []
    total_predictions = 0
    fraud_count = 0
    
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH)
            if len(df) > 0:
                total_predictions = len(df)
                fraud_count = (df['prediction'] == 'Fraud').sum()
                # Get last 5 records for preview
                recent_records = df.sort_values(by='timestamp', ascending=False).head(5).to_dict(orient='records')
        except Exception as e:
            print(f"Error reading log data: {e}")
    
    return render_template("index.html", 
                         active_page='home',
                         recent_records=recent_records,
                         total_predictions=total_predictions,
                         fraud_count=fraud_count)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data with proper validation
            name = request.form.get('name', '').strip()
            item = request.form.get('item', '').strip()
            city = request.form.get('city', '').strip()
            price_str = request.form.get('price', '0').strip()
            model_name = request.form.get('model', 'RandomForest').strip()
            
            # Validate required fields
            if not all([name, item, city, price_str, model_name]):
                return render_template("index.html", 
                    error="Please fill in all required fields",
                    active_page='predict'
                )
            
            # Convert price to float with error handling
            try:
                price = float(price_str)
                if price <= 0:
                    raise ValueError("Price must be positive")
            except (ValueError, TypeError):
                return render_template("index.html", 
                    error="Please enter a valid price amount",
                    active_page='predict'
                )
            
            # Generate temporal features
            temporal = generate_temporal_features()
            
            # Create feature dictionary
            features = {
                'name': name,
                'item': item,
                'city': city,
                'price': price,
                'quantity': 1,
                'payment_mode': 'credit_card',
                **temporal
            }
            
            # Convert to DataFrame
            input_data = pd.DataFrame([features])
            
            # Try to load model, fallback to simple logic if not found
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            if os.path.exists(model_path):
                try:
                    pipeline = joblib.load(model_path)
                    prediction = pipeline.predict(input_data)[0]
                    probabilities = pipeline.predict_proba(input_data)[0]
                    fraud_prob = probabilities[1]
                except Exception as e:
                    # Fallback if model fails
                    prediction = 1 if price > 2000 else 0  # Changed from 50000 to 2000
                    fraud_prob = 0.95 if price > 2000 else 0.05
            else:
                # More sophisticated fallback logic
                fraud_score = 0
                
                # Price factor (higher prices more suspicious)
                if price > 1000:
                    fraud_score += 0.3
                if price > 5000:
                    fraud_score += 0.2
                
                # Item factor (electronics more targeted)
                high_risk_items = ['laptop', 'phone', 'camera', 'watch']
                if item.lower() in high_risk_items:
                    fraud_score += 0.2
                
                # Time factor (late night transactions)
                if temporal['hour'] < 6 or temporal['hour'] > 22:
                    fraud_score += 0.1
                
                # City factor (some cities higher risk)
                high_risk_cities = ['mumbai', 'delhi']
                if city.lower() in high_risk_cities:
                    fraud_score += 0.1
                
                prediction = 1 if fraud_score > 0.4 else 0
                fraud_prob = min(fraud_score, 0.95)
            
            result = 'Fraud' if prediction == 1 else 'Not-Fraud'
            
            # Store result
            result_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'name': name,
                'item': item,
                'city': city,
                'price': price,
                'model': model_name,
                'prediction': result,
                'probability': f"{fraud_prob:.2%}",
                'feature_hash': generate_feature_hash(features)
            }
            
            session['last_prediction'] = result_entry
            
            # Log the prediction
            log_entry = pd.DataFrame([{
                'timestamp': result_entry['timestamp'],
                'name': name,
                'item': item,
                'city': city,
                'price': price,
                'model': model_name,
                'prediction': result,
                'probability': fraud_prob
            }])
            
            if os.path.exists(LOG_PATH):
                df = pd.read_csv(LOG_PATH)
                df = pd.concat([df, log_entry], ignore_index=True)
            else:
                df = log_entry
            
            df.to_csv(LOG_PATH, index=False)
            
            # Return to same page with result instead of redirecting
            performance = load_performance_metrics()
            return render_template("index.html", 
                                 active_page='predict', 
                                 performance=performance,
                                 result=result_entry)
            
        except Exception as e:
            return render_template("index.html", 
                error=f"Prediction error: {str(e)}",
                active_page='predict'
            )
    
    # Load performance metrics for model selection info
    performance = load_performance_metrics()
    return render_template("index.html", active_page='predict', performance=performance)

@app.route('/prediction-result')
def prediction_result():
    result = session.get('last_prediction')
    if not result:
        return redirect("/predict")
    return render_template("prediction_result.html", result=result)

@app.route('/about')
def about():
    # Load model metadata and performance
    performance = load_performance_metrics()
    return render_template("about.html", 
                           active_page='about',
                           metadata=MODEL_METADATA,
                           performance=performance)

@app.route('/history')
def history():
    print("DEBUG: History route called")
    print(f"DEBUG: LOG_PATH = {LOG_PATH}")
    print(f"DEBUG: File exists = {os.path.exists(LOG_PATH)}")
    
    if not os.path.exists(LOG_PATH):
        print("DEBUG: No log file found, returning empty records")
        return render_template("history.html", records=[], active_page='history')
    
    try:
        df = pd.read_csv(LOG_PATH)
        print(f"DEBUG: Found {len(df)} records in history")
        print(f"DEBUG: Columns: {df.columns.tolist()}")
        
        if len(df) > 0:
            print(f"DEBUG: First record: {df.iloc[0].to_dict()}")
        
        if len(df) == 0:
            print("DEBUG: Log file is empty")
            return render_template("history.html", records=[], active_page='history')
        
        df = df.sort_values(by='timestamp', ascending=False)
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        all_records = df.to_dict(orient='records')
        total_records = len(all_records)
        
        # Calculate pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_records = all_records[start_idx:end_idx]
        
        # Calculate pagination info
        total_pages = (total_records + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages
        
        print(f"DEBUG: Returning {len(paginated_records)} records to template (page {page} of {total_pages})")
        
        return render_template("history.html", 
                               records=paginated_records, 
                               active_page='history',
                               page=page,
                               total_pages=total_pages,
                               has_prev=has_prev,
                               has_next=has_next,
                               total_records=total_records)
    except Exception as e:
        print(f"DEBUG: Error reading log file: {e}")
        return render_template("history.html", records=[], active_page='history')

@app.route('/upload', methods=['GET'])
def upload_page():
    # Get total predictions count
    total_predictions = 0
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH)
            total_predictions = len(df)
        except Exception as e:
            print(f"Error reading log for predictions count: {e}")
    
    return render_template("upload.html", 
                          active_page='upload',
                          total_predictions=total_predictions)

@app.route('/upload', methods=['POST'])
def upload():
    if 'dataset' not in request.files:
        return render_template("upload.html", 
            error="No file selected",
            active_page='upload'
        )
    
    file = request.files['dataset']
    if file.filename == '':
        return render_template("upload.html", 
            error="No file selected",
            active_page='upload'
        )
    
    if file:
        # Save file
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        
        # Process file
        try:
            df = pd.read_csv(path)
            
            # Add required columns if missing
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if 'model' not in df.columns:
                df['model'] = 'RandomForest'
            if 'prediction' not in df.columns:
                # Add temporal features
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['timestamp'] = df['timestamp'].fillna(pd.Timestamp.now())
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.day_name()
                df['month'] = df['timestamp'].dt.month_name()
                
                # Predict using fallback logic
                df['prediction'] = df['price'].apply(lambda x: 'Fraud' if x > 2000 else 'Not-Fraud')
                df['probability'] = df['price'].apply(lambda x: 0.95 if x > 2000 else 0.05)
            
            # Save to log
            if os.path.exists(LOG_PATH):
                existing = pd.read_csv(LOG_PATH)
                df = pd.concat([existing, df], ignore_index=True)
            
            df.to_csv(LOG_PATH, index=False)
            
            return redirect("/visualize")
        except Exception as e:
            return render_template("upload.html", 
                error=f"Error processing file: {str(e)}",
                active_page='upload'
            )
    
    return render_template("upload.html", 
        error="Upload failed",
        active_page='upload'
    )

@app.route('/visualize')
def visualize():
    print("DEBUG: Visualize route called")
    print(f"DEBUG: LOG_PATH = {LOG_PATH}")
    print(f"DEBUG: File exists = {os.path.exists(LOG_PATH)}")
    
    if not os.path.exists(LOG_PATH):
        print("DEBUG: No log file found")
        return render_template("visualize.html", 
                               charts={},
                               records=[],
                               total=0,
                               fraud_count=0,
                               fraud_rate="0%",
                               active_page='visualize')
    
    try:
        df = pd.read_csv(LOG_PATH)
        print(f"DEBUG: Found {len(df)} records in log")
        print(f"DEBUG: Columns: {df.columns.tolist()}")
        
        if len(df) == 0:
            print("DEBUG: Log file is empty")
            return render_template("visualize.html", 
                                   charts={},
                                   records=[],
                                   total=0,
                                   fraud_count=0,
                                   fraud_rate="0%",
                                   active_page='visualize')
        
        # Calculate statistics
        total_transactions = len(df)
        fraud_count = (df['prediction'] == 'Fraud').sum()
        fraud_rate = f"{(fraud_count / total_transactions * 100):.1f}%" if total_transactions > 0 else "0%"
        
        print(f"DEBUG: Stats - Total: {total_transactions}, Fraud: {fraud_count}, Rate: {fraud_rate}")
        
        # Generate charts
        charts = {}
        
        def encode_chart(fig, filename=None):
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            if filename:
                fig.savefig(f'static/charts/{filename}.png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            chart_data = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)
            return chart_data
        
        # 1. Fraud Distribution Pie Chart
        try:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            fraud_counts = df['prediction'].value_counts()
            
            colors = ['#27ae60' if x == 'Not-Fraud' else '#e74c3c' for x in fraud_counts.index]
            explode = [0.1 if x == 'Fraud' else 0 for x in fraud_counts.index]
            
            ax1.pie(fraud_counts.values, labels=fraud_counts.index, autopct='%1.1f%%', 
                   colors=colors, explode=explode, shadow=True, startangle=90)
            ax1.set_title('Fraud Distribution', fontsize=14, fontweight='bold')
            charts['pie'] = encode_chart(fig1, 'fraud_distribution')
            print("DEBUG: Created pie chart")
        except Exception as e:
            print(f"Error creating pie chart: {e}")
        
        # 2. Price Distribution
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            fraud_prices = df[df['prediction'] == 'Fraud']['price'].dropna()
            safe_prices = df[df['prediction'] == 'Not-Fraud']['price'].dropna()
            
            ax2.hist([safe_prices, fraud_prices], bins=20, alpha=0.7, 
                    label=['Safe', 'Fraud'], color=['#27ae60', '#e74c3c'])
            ax2.set_xlabel('Price ($)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Price Distribution by Fraud Status', fontsize=14, fontweight='bold')
            ax2.legend()
            charts['price_dist'] = encode_chart(fig2, 'price_distribution')
            print("DEBUG: Created price distribution chart")
        except Exception as e:
            print(f"Error creating price distribution chart: {e}")
        
        # 3. Top Items
        try:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            top_items = df['item'].value_counts().head(10)
            
            if len(top_items) > 0:
                bars = ax3.barh(range(len(top_items)), top_items.values)
                ax3.set_yticks(range(len(top_items)))
                ax3.set_yticklabels(top_items.index)
                ax3.set_xlabel('Transaction Count')
                ax3.set_title('Top 10 Items by Transaction Count', fontsize=14, fontweight='bold')
                
                # Color bars
                for i, bar in enumerate(bars):
                    bar.set_color('#3498db')
                
                charts['top_items'] = encode_chart(fig3, 'top_items')
                print("DEBUG: Created top items chart")
        except Exception as e:
            print(f"Error creating top items chart: {e}")
        
        # 4. City Distribution
        try:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            city_counts = df['city'].value_counts().head(10)
            
            if len(city_counts) > 0:
                bars = ax4.bar(range(len(city_counts)), city_counts.values)
                ax4.set_xticks(range(len(city_counts)))
                ax4.set_xticklabels(city_counts.index, rotation=45, ha='right')
                ax4.set_ylabel('Transaction Count')
                ax4.set_title('Top 10 Cities by Transaction Count', fontsize=14, fontweight='bold')
                
                # Color bars
                for bar in bars:
                    bar.set_color('#9b59b6')
                
                charts['city_dist'] = encode_chart(fig4, 'city_distribution')
                print("DEBUG: Created city distribution chart")
        except Exception as e:
            print(f"Error creating city distribution chart: {e}")
        
        print(f"DEBUG: Created {len(charts)} charts")
        
        # Get recent records for display
        recent_records = df.sort_values(by='timestamp', ascending=False).head(10).to_dict(orient='records')
        
        return render_template("visualize.html", 
                               charts=charts,
                               records=recent_records,
                               total=total_transactions,
                               fraud_count=fraud_count,
                               fraud_rate=fraud_rate,
                               total_records=total_transactions,
                               active_page='visualize')
                               
    except Exception as e:
        print(f"Error in visualize route: {e}")
        import traceback
        traceback.print_exc()
        return render_template("visualize.html", 
                               charts={},
                               records=[],
                               total=0,
                               fraud_count=0,
                               fraud_rate="0%",
                               error=f"Error loading visualization: {str(e)}",
                               active_page='visualize')

@app.route('/model-performance')
def model_performance():
    performance = load_performance_metrics()
    if not performance:
        return render_template("model_performance.html", 
                               error="No performance data available",
                               active_page='performance')
    
    # Prepare model data
    models = []
    for name, metrics in performance.items():
        models.append({
            'name': name,
            'roc_auc': metrics['roc_auc'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': metrics['accuracy']
        })
    
    # Sort by ROC AUC
    models.sort(key=lambda x: x['roc_auc'], reverse=True)
    
    return render_template("model_performance.html", 
                           models=models,
                           metadata=MODEL_METADATA,
                           active_page='performance')

@app.route('/download')
def download():
    if os.path.exists(LOG_PATH):
        return send_file(
            LOG_PATH,
            as_attachment=True,
            download_name=f'fraud_predictions_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    return render_template("history.html", 
                           error="No data available for download",
                           active_page='history')

@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        # Clear transaction history log
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
            print("✅ Cleared transaction history")
        
        # Clear uploaded files
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("✅ Cleared uploaded files")
        
        # Clear generated charts
        charts_folder = 'static/charts'
        if os.path.exists(charts_folder):
            for filename in os.listdir(charts_folder):
                file_path = os.path.join(charts_folder, filename)
                if os.path.isfile(file_path) and filename.endswith('.png'):
                    os.remove(file_path)
            print("✅ Cleared generated charts")
        
        # Clear session data
        session.clear()
        
        return redirect(request.referrer or "/")
        
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        return redirect(request.referrer or "/")

@app.route('/download-sample/<sample_type>')
def download_sample(sample_type):
    """Download sample datasets"""
    try:
        if sample_type == 'ecommerce':
            filename = 'ecommerce_sample.csv'
        elif sample_type == 'retail':
            filename = 'retail_sample.csv'
        else:
            return "Sample type not found", 404
        
        # Check if file exists
        if os.path.exists(filename):
            return send_file(
                filename,
                as_attachment=True,
                download_name=filename,
                mimetype='text/csv'
            )
        else:
            return f"Sample file {filename} not found", 404
            
    except Exception as e:
        print(f"Error downloading sample: {e}")
        return f"Error: {str(e)}", 500

@app.route('/upload-stats')
def upload_stats():
    """Get upload statistics"""
    try:
        stats = {
            'total_files': 0,
            'total_records': 0,
            'last_upload': None
        }
        
        if os.path.exists(UPLOAD_FOLDER):
            files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
            stats['total_files'] = len(files)
            
            if files:
                # Get most recent file
                files_with_time = [(f, os.path.getctime(os.path.join(UPLOAD_FOLDER, f))) for f in files]
                latest_file = max(files_with_time, key=lambda x: x[1])
                stats['last_upload'] = datetime.fromtimestamp(latest_file[1]).strftime("%Y-%m-%d %H:%M:%S")
                
                # Count total records
                for file in files:
                    try:
                        df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file))
                        stats['total_records'] += len(df)
                    except:
                        pass
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Error getting upload stats: {e}")
        return jsonify({
            'total_files': 0,
            'total_records': 0,
            'last_upload': None
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
