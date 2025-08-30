#!/usr/bin/env python3
"""
Example usage of the Heart Disease ML Pipeline deployment system.

This script demonstrates different ways to use the deployment system
for various scenarios.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deployment.ngrok_setup import NgrokManager
from deployment.deploy import DeploymentManager


def example_basic_deployment():
    """Example: Basic deployment with default settings."""
    print("Example 1: Basic Deployment")
    print("-" * 30)
    
    # Create deployment manager
    manager = DeploymentManager(
        app_path="ui/streamlit_app.py",
        port=8501
    )
    
    print("This would deploy the app with:")
    print("- Default port: 8501")
    print("- No authentication token (free tier)")
    print("- Automatic monitoring enabled")
    print("- Health checking enabled")
    print()


def example_advanced_deployment():
    """Example: Advanced deployment with custom settings."""
    print("Example 2: Advanced Deployment")
    print("-" * 30)
    
    # Create deployment manager with custom settings
    manager = DeploymentManager(
        app_path="ui/streamlit_app.py",
        port=8502,
        auth_token="your_ngrok_token_here"
    )
    
    print("This would deploy the app with:")
    print("- Custom port: 8502")
    print("- Ngrok authentication token")
    print("- Better rate limits and features")
    print("- Custom subdomain support")
    print()


def example_ngrok_only():
    """Example: Using NgrokManager directly."""
    print("Example 3: Direct Ngrok Management")
    print("-" * 30)
    
    # Create Ngrok manager
    ngrok = NgrokManager(port=8501, auth_token="your_token")
    
    print("Direct Ngrok usage allows:")
    print("- Fine-grained tunnel control")
    print("- Custom tunnel configuration")
    print("- Manual process management")
    print("- Advanced monitoring")
    print()


def example_deployment_workflow():
    """Example: Complete deployment workflow."""
    print("Example 4: Complete Deployment Workflow")
    print("-" * 40)
    
    print("1. Install dependencies:")
    print("   pip install -r deployment/requirements.txt")
    print()
    
    print("2. Basic deployment:")
    print("   python deployment/deploy.py")
    print()
    
    print("3. Advanced deployment:")
    print("   python deployment/deploy.py \\")
    print("     --port 8502 \\")
    print("     --auth-token YOUR_TOKEN \\")
    print("     --subdomain my-heart-app")
    print()
    
    print("4. Monitor deployment:")
    print("   - Automatic monitoring is enabled by default")
    print("   - Check logs in deployment/deployment_log.json")
    print("   - Health checks run every 30 seconds")
    print()
    
    print("5. Stop deployment:")
    print("   - Press Ctrl+C to gracefully shutdown")
    print("   - All processes will be cleaned up automatically")
    print()


def example_troubleshooting():
    """Example: Common troubleshooting scenarios."""
    print("Example 5: Troubleshooting Common Issues")
    print("-" * 40)
    
    print("Issue 1: Port already in use")
    print("Solution: python deployment/deploy.py --port 8502")
    print()
    
    print("Issue 2: Ngrok installation fails")
    print("Solution: Manual installation or check internet connection")
    print()
    
    print("Issue 3: Tunnel connection fails")
    print("Solution: Use authentication token or check firewall")
    print()
    
    print("Issue 4: Application not accessible")
    print("Solution: Check local app first, then tunnel status")
    print()


def example_production_considerations():
    """Example: Production deployment considerations."""
    print("Example 6: Production Considerations")
    print("-" * 35)
    
    print("For production use, consider:")
    print("- Paid Ngrok plan for better performance")
    print("- Custom domain names")
    print("- Authentication in your Streamlit app")
    print("- HTTPS enforcement")
    print("- Rate limiting and monitoring")
    print("- Data privacy and security")
    print()
    
    print("Alternative production options:")
    print("- Streamlit Cloud")
    print("- Heroku deployment")
    print("- AWS/GCP/Azure cloud hosting")
    print("- Docker containerization")
    print()


def main():
    """Run all examples."""
    print("🚀 Heart Disease ML Pipeline - Deployment Examples")
    print("=" * 55)
    print()
    
    example_basic_deployment()
    example_advanced_deployment()
    example_ngrok_only()
    example_deployment_workflow()
    example_troubleshooting()
    example_production_considerations()
    
    print("📚 For more information:")
    print("- Read deployment/README.md")
    print("- Check deployment/test_deployment.py")
    print("- Review the source code in deployment/")
    print()
    
    print("🎯 Ready to deploy? Run:")
    print("python deployment/deploy.py")


if __name__ == "__main__":
    main()