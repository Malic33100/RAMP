 RAMP - Risk Assessment & Market Prediction

 Prerequisites
- Python 3.13 or newer
- 50MB free disk space

 Installation Steps
1. Download folder

 2. Create Virtual Environment
```bash
python -m venv ramp_env




3.Initialize virtual environment
   ramp_env/Scripts/activate (windows in bash terminal)
   source ramp_env/bin/activate (mac/linux)


4. install requirements
pip install -r requirements.txt

5. run project 
The submission includes a pre-populated database with historical data. Simply run:
python cli/ramp_cli.py



Usage Examples

View all RAMP scores: python cli/ramp_cli.py
Check specific stock: python cli/ramp_cli.py --stock AAPL
View model accuracy: python cli/ramp_cli.py --accuracy