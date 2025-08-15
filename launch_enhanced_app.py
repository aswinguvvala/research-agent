#!/usr/bin/env python3
"""
Launch script for Enhanced AI Research Agent Pro
Provides easy startup with environment validation and helpful guidance.
"""

import os
import sys
import subprocess
import importlib
import streamlit as st
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'openai', 'arxiv', 
        'wikipedia', 'scholarly', 'newspaper3k', 'numpy', 
        'networkx', 'matplotlib', 'requests', 'beautifulsoup4'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_api_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        # Check Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                return True
        except:
            pass
        return False
    
    return True

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🚀 AI Research Agent Pro - Enhanced UI             ║
    ║                                                              ║
    ║        Next-generation autonomous research intelligence      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_setup_instructions():
    """Print setup instructions"""
    print("\n📋 SETUP INSTRUCTIONS:")
    print("=" * 50)
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        print("\n   Or install all requirements:")
        print("   pip install -r requirements-enhanced.txt")
    else:
        print("✅ All required packages are installed")
    
    # Check API key
    if not check_api_key():
        print("\n❌ OpenAI API Key not configured")
        print("\n🔑 Set up your API key:")
        print("   Option 1: Environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("\n   Option 2: Streamlit secrets file")
        print("   Create .streamlit/secrets.toml with:")
        print("   OPENAI_API_KEY = 'your-api-key-here'")
    else:
        print("✅ OpenAI API Key is configured")

def launch_app():
    """Launch the enhanced Streamlit app"""
    app_file = Path(__file__).parent / "enhanced_self_initiated_app.py"
    
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        print("   Make sure you're in the correct directory")
        return False
    
    print(f"\n🚀 Launching Enhanced AI Research Agent Pro...")
    print(f"   App file: {app_file}")
    print("\n💡 The app will open in your default web browser")
    print("   If it doesn't open automatically, go to: http://localhost:8501")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--theme.base", "light",
            "--theme.primaryColor", "#3b82f6",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f8fafc",
            "--theme.textColor", "#1f2937"
        ])
        return True
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check if we should just launch or show setup
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        launch_app()
        return
    
    print_setup_instructions()
    
    # Check if everything is ready
    missing_packages = check_requirements()
    api_key_configured = check_api_key()
    
    if missing_packages or not api_key_configured:
        print("\n⚠️  Setup incomplete. Please fix the issues above before launching.")
        print("\n💡 To launch anyway (for testing), use: python launch_enhanced_app.py --force")
        return
    
    print("\n🎉 Setup complete! Ready to launch.")
    
    # Ask user if they want to launch
    try:
        response = input("\n🚀 Launch Enhanced AI Research Agent Pro? (y/n): ").strip().lower()
        if response in ['y', 'yes', '']:
            launch_app()
        else:
            print("👋 Launch cancelled. Run this script again when ready!")
    except KeyboardInterrupt:
        print("\n\n👋 Launch cancelled.")

if __name__ == "__main__":
    main()
