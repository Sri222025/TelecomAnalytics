 Project Structure
telecom_analytics/
│
├── 📄 app.py                          # Main Streamlit application (entry point)
│   └── Multi-page interface with navigation
│
├── 📊 Core Processing Modules
│   ├── file_processor.py              # Upload & parse Excel/CSV files
│   ├── relationship_detector.py       # Auto-detect data relationships
│   ├── data_merger.py                 # Merge datasets based on relationships
│   └── telecom_metrics.py             # Calculate telecom KPIs (ARPU, MOU, etc.)
│
├── 🤖 AI & Analytics Modules
│   ├── anomaly_detector.py            # AI-powered anomaly detection (MAIN USP)
│   └── insights_generator.py          # Generate natural language insights
│
├── 📈 Visualization & Export
│   ├── visualizations.py              # Plotly charts and graphs
│   ├── export_manager.py              # Export to CSV, Excel, PDF
│   └── config_manager.py              # Save/load configurations
│
├── ⚙️ Configuration
│   ├── requirements.txt               # Python dependencies
│   └── .streamlit/
│       └── config.toml                # Streamlit configuration
│
└── 📖 Documentation
    ├── README.md                      # Comprehensive documentation
    ├── QUICKSTART.md                  # 5-minute setup guide
    ├── DEPLOYMENT_GUIDE.md            # Detailed deployment instructions
    └── PROJECT_STRUCTURE.md           # This file
📄 File Descriptions
Main Application
app.py (46KB)

Entry point for the Streamlit application
10 pages: Home, Upload, Relationships, Alerts, Subscriber/Device/Usage/Regional Analytics, Comparisons, Export
Session state management
Navigation and UI layout
Core Processing
file_processor.py (4KB)

Multi-file upload handling
Excel multi-sheet parsing
CSV support
Data type detection
Preview generation
relationship_detector.py (8KB)

Auto-detect common columns across files
Pattern matching for linking keys
Confidence scoring
Validation logic
data_merger.py (7KB)

Merge datasets on relationships
Handle complex joins
Prevent column conflicts
Concatenation fallback
telecom_metrics.py (11KB)

Calculate subscriber metrics
Usage analytics (call volume, duration)
Device format metrics
Regional performance
Temporal analysis
Data quality checks
AI & Analytics
anomaly_detector.py (15KB) ⭐ MAIN USP

Data quality issue detection
Usage anomaly detection
Subscriber pattern analysis
Regional performance anomalies
Device usage anomalies
Statistical outlier detection
Temporal anomaly detection
Severity classification (Critical/Warning/Info)
insights_generator.py (13KB)

Executive summary generation
Key findings extraction
Actionable recommendations
Trend identification
Priority ranking
Visualization & Export
visualizations.py (5KB)

Plotly chart generation
Distribution charts
Trend lines
Regional heatmaps
Device comparisons
Correlation matrices
export_manager.py (3KB)

CSV export
Multi-sheet Excel export
Text report generation
Summary dataframe creation
config_manager.py (1KB)

Save/load relationships
User preferences
JSON configuration
Configuration
requirements.txt

streamlit==1.31.0       # Web framework
pandas==2.1.4           # Data processing
numpy==1.26.3           # Numerical computing
plotly==5.18.0          # Interactive visualizations
openpyxl==3.1.2         # Excel file handling
scipy==1.11.4           # Statistical functions
python-dateutil==2.8.2  # Date parsing
.streamlit/config.toml

Theme colors
Upload size limits
Server configuration
🔄 Data Flow
1. Upload Files (Excel/CSV)
        ↓
2. File Processor → Parse all sheets
        ↓
3. Relationship Detector → Find common columns
        ↓
4. Data Merger → Join datasets
        ↓
5. Telecom Metrics → Calculate KPIs
        ↓
6. Anomaly Detector → Flag issues (AI)
        ↓
7. Insights Generator → Create summaries
        ↓
8. Visualizations → Generate dashboards
        ↓
9. Export Manager → Download results
🧩 Module Dependencies
app.py
├── file_processor
├── relationship_detector
├── data_merger
├── telecom_metrics
├── anomaly_detector
├── insights_generator
├── visualizations
├── export_manager
└── config_manager

anomaly_detector
└── telecom_metrics (uses metrics for analysis)

insights_generator
├── telecom_metrics
└── anomaly_detector (uses anomalies for insights)

data_merger
└── file_processor (uses processed data)
📦 Total Size
Code: ~130 KB
Documentation: ~25 KB
Total Package: ~155 KB
🔧 Customization Points
Easy Customizations
Anomaly Thresholds: Edit anomaly_detector.py lines 8-14
Color Scheme: Edit visualizations.py line 7
Metric Calculations: Add to telecom_metrics.py
Page Layout: Modify app.py sections
Advanced Customizations
Add new dashboard pages in app.py
Create custom visualizations in visualizations.py
Implement new detection algorithms in anomaly_detector.py
Add export formats in export_manager.py
🎯 Key Features by File
Feature	Primary File	Supporting Files
Multi-file upload	file_processor.py	app.py
Auto-relationship detection	relationship_detector.py	data_merger.py
AI anomaly detection ⭐	anomaly_detector.py	telecom_metrics.py
Natural language insights	insights_generator.py	anomaly_detector.py
Interactive dashboards	app.py	visualizations.py
KPI calculations	telecom_metrics.py	-
Data export	export_manager.py	app.py
💡 Usage Example
Copy# Example: How modules work together

# 1. Upload files
processor = FileProcessor()
processed_data = processor.process_files(uploaded_files)

# 2. Detect relationships
detector = RelationshipDetector()
relationships = detector.detect_relationships(processed_data)

# 3. Merge data
merger = DataMerger()
merged_data = merger.merge_data(processed_data, relationships)

# 4. Calculate metrics
metrics_calc = TelecomMetrics()
metrics = metrics_calc.calculate_metrics(merged_data)

# 5. Detect anomalies (MAIN VALUE)
anomaly_detector = AnomalyDetector()
anomalies = anomaly_detector.detect_anomalies(merged_data, metrics)

# 6. Generate insights
insights_gen = InsightsGenerator()
insights = insights_gen.generate_insights(merged_data, metrics, anomalies)
