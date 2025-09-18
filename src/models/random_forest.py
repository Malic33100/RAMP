#!/usr/bin/env python3
"""
random_forest.py - RAMP Random Forest Volatility Prediction Model
Generates RAMP scores (0-100) where 100 = highest predicted volatility
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')

class RAMPVolatilityModel:
    def __init__(self, db_path="ramp_database.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self):
        """Load and prepare data from SQLite database"""
        print("📊 Loading data from database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load stock data - exclude indices for now
        query = """
        SELECT symbol, date, open, high, low, close, volume 
        FROM stock_data 
        WHERE symbol NOT LIKE 'INDEX_%'
        ORDER BY symbol, date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"   ✅ Loaded {len(df):,} stock records")
        print(f"   ✅ {df['symbol'].nunique()} unique stocks")
        print(f"   ✅ Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def engineer_features(self, df):
        """Create features for volatility prediction"""
        print("🔧 Engineering features...")
        
        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        
        features_df = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy().sort_values('date')
            
            if len(symbol_data) < 30:  # Need enough data for features
                continue
            
            # Basic price features
            symbol_data['daily_return'] = symbol_data['close'].pct_change()
            symbol_data['high_low_pct'] = (symbol_data['high'] - symbol_data['low']) / symbol_data['close']
            symbol_data['open_close_pct'] = (symbol_data['close'] - symbol_data['open']) / symbol_data['open']
            
            # Moving averages
            symbol_data['sma_5'] = symbol_data['close'].rolling(5).mean()
            symbol_data['sma_10'] = symbol_data['close'].rolling(10).mean()
            symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
            
            # Price relative to moving averages
            symbol_data['price_sma5_ratio'] = symbol_data['close'] / symbol_data['sma_5']
            symbol_data['price_sma10_ratio'] = symbol_data['close'] / symbol_data['sma_10']
            symbol_data['price_sma20_ratio'] = symbol_data['close'] / symbol_data['sma_20']
            
            # Volatility measures (historical)
            symbol_data['volatility_5d'] = symbol_data['daily_return'].rolling(5).std()
            symbol_data['volatility_10d'] = symbol_data['daily_return'].rolling(10).std()
            symbol_data['volatility_20d'] = symbol_data['daily_return'].rolling(20).std()
            
            # Volume features
            symbol_data['volume_sma_10'] = symbol_data['volume'].rolling(10).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma_10']
            
            # Rate of Change (momentum)
            symbol_data['roc_5d'] = symbol_data['close'].pct_change(5)
            symbol_data['roc_10d'] = symbol_data['close'].pct_change(10)
            
            # TARGET VARIABLES: Forward volatility (what we want to predict)
            # 1-day forward volatility
            symbol_data['target_volatility_1d'] = symbol_data['daily_return'].shift(-1).abs()
            
            # 3-day forward volatility (rolling standard deviation of next 3 days)
            future_returns = symbol_data['daily_return'].shift(-1)
            for i in range(2, 4):
                future_returns += symbol_data['daily_return'].shift(-i)
            symbol_data['target_volatility_3d'] = future_returns.rolling(3).std().shift(-2)
            
            features_df.append(symbol_data)
        
        # Combine all symbols
        final_df = pd.concat(features_df, ignore_index=True)
        
        print(f"   ✅ Created features for {final_df['symbol'].nunique()} stocks")
        print(f"   ✅ {len(final_df):,} total feature rows")
        
        return final_df
    
    def prepare_training_data(self, df, target_col='target_volatility_1d'):
        """Prepare data for model training"""
        print(f"🎯 Preparing training data (target: {target_col})...")
        
        # Select features (exclude non-feature columns)
        feature_cols = [
            'daily_return', 'high_low_pct', 'open_close_pct',
            'price_sma5_ratio', 'price_sma10_ratio', 'price_sma20_ratio',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'volume_ratio', 'roc_5d', 'roc_10d'
        ]
        
        # Filter out rows with missing values
        model_data = df[feature_cols + [target_col, 'symbol', 'date']].dropna()
        
        print(f"   ✅ {len(model_data):,} complete records for training")
        
        X = model_data[feature_cols]
        y = model_data[target_col]
        
        self.feature_names = feature_cols
        
        return X, y, model_data[['symbol', 'date']]
    
    def train_model(self, X, y, test_size=0.2):
        """Train Random Forest model"""
        print("🌲 Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"   ✅ Training R² Score: {train_score:.4f}")
        print(f"   ✅ Testing R² Score: {test_score:.4f}")
        print(f"   ✅ RMSE: {rmse:.6f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 5 Most Important Features:")
        for i, row in feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    
    def generate_ramp_scores(self, df, target_col='target_volatility_1d'):
        """Generate RAMP scores for current data"""
        print("📊 Generating RAMP scores...")
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get most recent data for each stock
        latest_data = df.groupby('symbol').last().reset_index()
        
        # Prepare features
        X, _, metadata = self.prepare_training_data(
            pd.concat([df, latest_data]), target_col
        )
        
        # Filter to get only latest predictions
        latest_mask = metadata.groupby('symbol')['date'].transform('max') == metadata['date']
        X_latest = X[latest_mask]
        metadata_latest = metadata[latest_mask]
        
        if len(X_latest) == 0:
            print("   ❌ No recent data available for predictions")
            return pd.DataFrame()
        
        # Scale features and predict
        X_scaled = self.scaler.transform(X_latest)
        predictions = self.model.predict(X_scaled)
        
        # Convert to RAMP scores (0-100 scale)
        # Higher volatility = higher RAMP score
        min_pred = predictions.min()
        max_pred = predictions.max()
        
        if max_pred > min_pred:
            ramp_scores = 100 * (predictions - min_pred) / (max_pred - min_pred)
        else:
            ramp_scores = np.full(len(predictions), 50)  # Default to 50 if no variation
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'symbol': metadata_latest['symbol'].values,
            'date': metadata_latest['date'].values,
            'predicted_volatility': predictions,
            'ramp_score': ramp_scores.round(1)
        }).sort_values('ramp_score', ascending=False)
        
        print(f"   ✅ Generated RAMP scores for {len(results_df)} stocks")
        
        return results_df
    
    def save_model(self, filepath="models/ramp_volatility_model.joblib"):
        """Save trained model and scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath="models/ramp_volatility_model.joblib"):
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"📥 Model loaded from {filepath}")


def analyze_industries(ramp_scores):
    """Analyze which industries have highest/lowest average RAMP scores"""
    
    # Simple industry mapping based on stock symbols
    industry_map = {
        # Tech
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
        'META': 'Technology', 'NVDA': 'Technology', 'TSLA': 'Technology', 'ORCL': 'Technology',
        'CRM': 'Technology', 'ADBE': 'Technology', 'INTC': 'Technology', 'AMD': 'Technology',
        'AVGO': 'Technology', 'TXN': 'Technology', 'AMAT': 'Technology', 'LRCX': 'Technology',
        'QCOM': 'Technology', 'IBM': 'Technology', 'NOW': 'Technology', 'INTU': 'Technology',
        
        # Healthcare
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare',
        'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare', 'BMY': 'Healthcare',
        'LLY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'ISRG': 'Healthcare',
        'SYK': 'Healthcare', 'VRTX': 'Healthcare', 'REGN': 'Healthcare', 'BDX': 'Healthcare',
        'ZTS': 'Healthcare', 'CI': 'Healthcare', 'CVS': 'Healthcare', 'ELV': 'Healthcare',
        
        # Financial
        'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'MS': 'Financial',
        'GS': 'Financial', 'AXP': 'Financial', 'C': 'Financial', 'SCHW': 'Financial',
        'BLK': 'Financial', 'SPGI': 'Financial', 'MMC': 'Financial', 'CB': 'Financial',
        
        # Consumer
        'AMZN': 'Consumer', 'WMT': 'Consumer', 'PG': 'Consumer', 'HD': 'Consumer',
        'MCD': 'Consumer', 'COST': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer',
        'NKE': 'Consumer', 'SBUX': 'Consumer', 'TJX': 'Consumer', 'LOW': 'Consumer',
        'MDLZ': 'Consumer', 'TGT': 'Consumer',
        
        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        
        # Industrial
        'BA': 'Industrial', 'CAT': 'Industrial', 'DE': 'Industrial', 'HON': 'Industrial',
        'UPS': 'Industrial', 'RTX': 'Industrial', 'GE': 'Industrial', 'ETN': 'Industrial',
        
        # Other
        'V': 'Financial', 'MA': 'Financial', 'UNH': 'Healthcare', 'NEE': 'Utilities',
        'VZ': 'Telecom', 'T': 'Telecom', 'CMCSA': 'Media', 'NFLX': 'Media',
        'DIS': 'Media', 'PM': 'Consumer', 'MO': 'Consumer', 'TMUS': 'Telecom',
        'SO': 'Utilities', 'PLD': 'Real Estate', 'EQIX': 'Real Estate', 'BKNG': 'Consumer'
    }
    
    # Add industry column
    ramp_scores['industry'] = ramp_scores['symbol'].map(industry_map)
    ramp_scores['industry'] = ramp_scores['industry'].fillna('Other')
    
    # Calculate average RAMP score by industry
    industry_analysis = ramp_scores.groupby('industry')['ramp_score'].agg(['mean', 'count']).round(1)
    industry_analysis = industry_analysis.sort_values('mean', ascending=False)
    
    return industry_analysis, ramp_scores


def calculate_simple_accuracy(ramp_model, featured_data, ramp_scores):
    """Simple accuracy check: Did our top predictions actually happen?"""
    print("🎯 Calculating RAMP Prediction Accuracy...")
    
    # Get the most recent data with actual next-day volatility
    recent_data = featured_data.dropna(subset=['target_volatility_1d']).copy()
    
    if len(recent_data) < 50:
        print("❌ Not enough recent data for accuracy check")
        return None
    
    # Get latest date for each stock (to match our predictions)
    latest_data = recent_data.groupby('symbol').tail(1).copy()
    
    # Merge with our RAMP scores
    merged = latest_data.merge(ramp_scores, on='symbol', how='inner')
    
    if len(merged) < 20:
        print("❌ Not enough matching data for accuracy check")
        return None
    
    # Test HIGH volatility predictions (Top 10 RAMP scores)
    top_10 = merged.nlargest(10, 'ramp_score')
    high_vol_actual = (top_10['target_volatility_1d'] >= 0.01).sum()  # ≥1%
    high_vol_accuracy = (high_vol_actual / 10) * 100
    
    # Test LOW volatility predictions (Bottom 10 RAMP scores)  
    bottom_10 = merged.nsmallest(10, 'ramp_score')
    # Check if actual volatility is within 0.5% of our prediction
    prediction_error = abs(bottom_10['target_volatility_1d'] - bottom_10['predicted_volatility'])
    low_vol_accurate = (prediction_error <= 0.005).sum()  # Within 0.5%
    low_vol_accuracy = (low_vol_accurate / 10) * 100
    
    # Overall accuracy (combine both tests)
    total_correct = high_vol_actual + low_vol_accurate
    overall_accuracy = (total_correct / 20) * 100
    
    print(f"\n📊 RAMP ACCURACY RESULTS:")
    print("=" * 40)
    print(f"High Volatility Accuracy: {high_vol_accuracy:.0f}%")
    print(f"  ({high_vol_actual}/10 high RAMP stocks had ≥1% volatility)")
    print(f"Low Volatility Accuracy:  {low_vol_accuracy:.0f}%") 
    print(f"  ({low_vol_accurate}/10 low RAMP stocks within 0.5% of prediction)")
    print(f"Overall RAMP Accuracy:    {overall_accuracy:.0f}%")
    
    if overall_accuracy >= 70:
        print("✅ EXCELLENT accuracy!")
    elif overall_accuracy >= 60:
        print("✅ GOOD accuracy!")
    elif overall_accuracy >= 50:
        print("⚠️  FAIR accuracy (better than random)")
    else:
        print("❌ LOW accuracy - model needs improvement")
    
    return {
        'high_vol_accuracy': high_vol_accuracy,
        'low_vol_accuracy': low_vol_accuracy,
        'overall_accuracy': overall_accuracy
    }


def main():
    """Main execution function"""
    print("🚀 RAMP Volatility Model - Starting...")
    print("=" * 50)
    
    # Initialize model
    ramp_model = RAMPVolatilityModel()
    
    # Load and process data
    raw_data = ramp_model.load_data()
    featured_data = ramp_model.engineer_features(raw_data)
    
    # Prepare training data (using 1-day volatility target)
    X, y, metadata = ramp_model.prepare_training_data(
        featured_data, target_col='target_volatility_1d'
    )
    
    # Train model
    feature_importance = ramp_model.train_model(X, y)
    
    # Generate RAMP scores
    ramp_scores = ramp_model.generate_ramp_scores(featured_data)
    
    # Calculate simple accuracy
    accuracy = calculate_simple_accuracy(ramp_model, featured_data, ramp_scores)
    
    # Generate RAMP scores
    initial_ramp_scores = ramp_model.generate_ramp_scores(featured_data)
    
    # Calculate simple accuracy (do this before industry analysis)
    accuracy = calculate_simple_accuracy(ramp_model, featured_data, initial_ramp_scores)
    
    # Analyze industries  
    industry_analysis, final_ramp_scores = analyze_industries(initial_ramp_scores)
    
    # Display industry analysis
    print(f"\n📊 INDUSTRY ANALYSIS:")
    print("=" * 50)
    highest_industry = industry_analysis.index[0]
    lowest_industry = industry_analysis.index[-1]
    print(f"🔴 Highest Volatility Industry: {highest_industry} (Avg RAMP: {industry_analysis.loc[highest_industry, 'mean']:.1f})")
    print(f"🟢 Lowest Volatility Industry: {lowest_industry} (Avg RAMP: {industry_analysis.loc[lowest_industry, 'mean']:.1f})")
    
    # Display top 10 highest volatility stocks
    print("\n🎯 TOP 10 HIGHEST RAMP SCORES (Most Volatile):")
    print("=" * 65)
    for i, row in final_ramp_scores.head(10).iterrows():
        vol_percent = row['predicted_volatility'] * 100
        print(f"{row['symbol']:>6} | RAMP Score: {row['ramp_score']:>5.1f} | Pred Vol: {vol_percent:>6.2f}%")
    
    # Display bottom 10 (least volatile)
    print("\n🔒 TOP 10 LOWEST RAMP SCORES (Least Volatile):")
    print("=" * 65)
    for i, row in final_ramp_scores.tail(10).iterrows():
        vol_percent = row['predicted_volatility'] * 100
        print(f"{row['symbol']:>6} | RAMP Score: {row['ramp_score']:>5.1f} | Pred Vol: {vol_percent:>6.2f}%")
    
    # Save model
    ramp_model.save_model()
    
    # Save results to CSV
    final_ramp_scores.to_csv('ramp_scores_latest.csv', index=False)
    print(f"\n💾 RAMP scores saved to ramp_scores_latest.csv")
    
    print("\n🎉 RAMP Model Complete! Your volatility prediction system is ready!")
    return ramp_model, final_ramp_scores

if __name__ == "__main__":
    model, scores = main()