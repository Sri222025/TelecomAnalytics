Telecom Analytics Platform
A powerful, AI-driven analytics platform for telecom data with automatic anomaly detection and intelligent insights generation.

🌟 Key Features
Core Capabilities
Dynamic Multi-File Processing: Upload any telecom data files (Excel with multiple worksheets, CSV)
Smart Data Linking: Automatically detects relationships across files and sheets
AI-Powered Anomaly Detection: Flags critical issues, unusual patterns, and data quality problems
Comprehensive Dashboards: Subscriber, usage, device format, and regional analytics
Natural Language Insights: AI-generated executive summaries and recommendations
Flexible Comparisons: Week-over-Week, Month-over-Month, Quarter-over-Quarter analysis
Easy Export: Download processed data, reports, and dashboards
Telecom-Specific Intelligence
Multiple Calling Formats: POTS (Landline), JioJoin App, STB Calling, AirFiber
Standard KPIs: ARPU, MOU, Churn Rate, Penetration, Activation Rate
Device Analytics: Track adoption across different calling methods
Regional Performance: Compare metrics across circles and regions
🚀 Quick Start
Prerequisites
GitHub account
Streamlit Cloud account (free) at streamlit.io
Deployment Steps
Step 1: Create GitHub Repository
Go to github.com and sign in
Click "New Repository"
Name it: telecom-analytics (or any name you prefer)
Set it to Public
Click "Create Repository"
Step 2: Upload Code to GitHub
On your repository page, click "uploading an existing file"
Drag and drop ALL these files:
app.py
file_processor.py
relationship_detector.py
data_merger.py
telecom_metrics.py
anomaly_detector.py
insights_generator.py
visualizations.py
export_manager.py
config_manager.py
requirements.txt
README.md
Click "Commit changes"
Step 3: Deploy on Streamlit Cloud
Go to share.streamlit.io
Sign in with your GitHub account
Click "New app"
Select:
Repository: your-username/telecom-analytics
Branch: main
Main file path: app.py
Click "Deploy"
Wait 2-3 minutes for deployment
Step 4: Access Your App
You'll get a URL like: https://your-username-telecom-analytics.streamlit.app
Share this URL with your team!
📖 How to Use
1. Upload Files
Go to 📤 Upload & Process page
Upload 3-4 Excel or CSV files (supports multiple worksheets)
Click "🔄 Process Files"
Wait for processing to complete
2. Review Relationships
Go to 🔗 Data Relationships page
Review auto-detected relationships between files
Add manual relationships if needed
Click "🔀 Merge Data Based on Relationships"
3. Check Alerts
Go to 🚨 Alerts & Anomalies page
Review critical issues flagged by AI
Read AI-generated insights and recommendations
This is the main value of the tool!
4. Explore Analytics
Navigate through different dashboard pages:

👥 Subscriber Analytics: Connection types, status, growth
📱 Device Analytics: POTS vs JioJoin vs STB usage
📞 Usage Analytics: Call volumes, duration, patterns
🗺️ Regional Performance: Geographic distribution and comparisons
📊 Comparisons: Period-over-period analysis
5. Export Results
Go to 💾 Export & Reports page
Download processed data as CSV or Excel
Download insights report as text file
Share with stakeholders
🎯 Use Cases
Weekly Operations Review
Upload weekly CDR and subscriber dumps
Compare with previous week
Check for anomalies
Take action on flagged issues
Monthly Performance Analysis
Upload monthly consolidated data
Generate comprehensive dashboards
Extract insights for management reports
Export formatted reports
Ad-hoc Investigations
Upload specific datasets for analysis
Use custom date range comparisons
Explore regional or device-specific patterns
Generate targeted recommendations
🔧 Configuration
Customizing Anomaly Thresholds
Edit anomaly_detector.py lines 8-14:

Copyself.thresholds = {
    'churn_spike': 0.2,  # 20% increase triggers alert
    'usage_drop': 0.3,   # 30% decrease triggers alert
    'zero_usage_threshold': 0.15,  # 15% zero usage triggers alert
    'outlier_zscore': 3,  # 3 standard deviations
    'missing_data_critical': 20,  # 20% missing data is critical
}
Adding Custom Metrics
Edit telecom_metrics.py to add domain-specific calculations.

📊 Sample Data Structure
The tool works with any structure, but typical telecom files include:

Subscriber Master
Customer_ID, Serial_Number, Fixed_Line_Number
Connection_Type (Fiber/AirFiber)
Region, Circle, Status
Activation_Date, Plan_Type
Usage Data (CDR)
Customer_ID, Call_Date, Duration
Call_Type (Local/STD/ISD)
Device_Type (POTS/JioJoin/STB)
Destination, Charges
Device Usage
Customer_ID, Device_Type
Usage_Date, Session_Count
Data_Consumed
🛠️ Troubleshooting
File Upload Issues
Error: "File too large"

Solution: Streamlit Cloud has 200MB limit. Split large files.
Error: "Cannot read Excel file"

Solution: Save as .xlsx (not .xls). Remove password protection.
Relationship Detection Issues
Problem: No relationships detected
Solution: Use manual relationship configuration
Check if columns have common values
Ensure column names are meaningful
Performance Issues
Problem: App is slow
Solution: Processing 1M+ records can take 2-3 minutes
Use date filtering to reduce data volume
Process during off-peak hours
🔒 Data Privacy
All data processing happens in Streamlit Cloud's secure environment
Data is NOT permanently stored (session-based only)
Files are deleted when session ends
For sensitive data, consider self-hosting on private infrastructure
🆘 Support
Common Questions
Q: Can I process more than 4 files? A: Yes! The tool supports any number of files. 3-4 is just a recommendation for performance.

Q: How do I save my relationship configuration? A: Currently stored in session. Future versions will add persistent config storage.

Q: Can I schedule automatic runs? A: Not in free Streamlit Cloud. Consider upgrading or self-hosting with cron jobs.

Q: Is my data secure? A: Data stays in your Streamlit session. It's not logged or persisted beyond the session.

🚀 Future Enhancements (Roadmap)
 Persistent configuration storage
 Scheduled report generation
 Email alert notifications
 More visualization types
 Machine learning predictions
 API integration for automated data feeds
 Multi-user authentication
 Custom theme support
📝 Version History
v1.0 (Current)
Initial release
Multi-file, multi-sheet processing
AI-powered anomaly detection
Comprehensive telecom KPIs
Interactive dashboards
Export functionality
👥 Credits
Built with:

Streamlit - Web framework
Pandas - Data processing
Plotly - Visualizations
SciPy - Statistical analysis
📄 License
This project is provided as-is for internal telecom analytics use.

Happy Analyzing! 📊📡

For issues or questions, create an issue in the GitHub repository
