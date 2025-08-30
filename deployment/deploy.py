#!/usr/bin/env python3
"""
One-Command Deployment Script

This script provides a simple interface to deploy the Heart Disease ML Pipeline
with Ngrok for public access. It handles Streamlit configuration, Ngrok setup,
and provides monitoring capabilities.
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deployment.ngrok_setup import NgrokManager, create_deployment_log, deployment_health_check
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamlitManager:
    """Manages Streamlit application lifecycle."""
    
    def __init__(self, app_path: str, port: int = 8501):
        """
        Initialize StreamlitManager.
        
        Args:
            app_path: Path to Streamlit application file
            port: Port to run Streamlit on
        """
        self.app_path = Path(app_path)
        self.port = port
        self.process = None
        
    def configure_streamlit_for_ngrok(self) -> None:
        """Configure Streamlit settings optimized for Ngrok deployment."""
        config_dir = Path.home() / ".streamlit"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "config.toml"
        
        config_content = f"""
[server]
port = {self.port}
enableCORS = false
enableXsrfProtection = false
enableWebsocketCompression = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info("Streamlit configured for Ngrok deployment")
    
    def start_streamlit(self) -> bool:
        """
        Start Streamlit application.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            if not self.app_path.exists():
                logger.error(f"Streamlit app not found: {self.app_path}")
                return False
            
            # Configure Streamlit
            self.configure_streamlit_for_ngrok()
            
            # Start Streamlit process
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                str(self.app_path),
                "--server.port", str(self.port),
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ]
            
            logger.info(f"Starting Streamlit application: {self.app_path}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=project_root
            )
            
            # Wait for Streamlit to start
            time.sleep(5)
            
            if self.process.poll() is None:
                logger.info(f"Streamlit started successfully on port {self.port}")
                return True
            else:
                logger.error("Streamlit failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Streamlit: {str(e)}")
            return False
    
    def stop_streamlit(self) -> None:
        """Stop Streamlit application."""
        if self.process:
            logger.info("Stopping Streamlit application...")
            self.process.terminate()
            
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Forcing Streamlit termination...")
                self.process.kill()
                self.process.wait()
            
            self.process = None
            logger.info("Streamlit stopped")


class DeploymentManager:
    """Manages complete deployment lifecycle."""
    
    def __init__(self, app_path: str, port: int = 8501, auth_token: Optional[str] = None):
        """
        Initialize DeploymentManager.
        
        Args:
            app_path: Path to Streamlit application
            port: Port for Streamlit application
            auth_token: Ngrok authentication token
        """
        self.streamlit_manager = StreamlitManager(app_path, port)
        self.ngrok_manager = NgrokManager(port, auth_token)
        self.is_running = False
        self.start_time = None
        
    def deploy(self, subdomain: Optional[str] = None) -> bool:
        """
        Deploy the application with Ngrok.
        
        Args:
            subdomain: Custom subdomain for Ngrok (requires paid plan)
            
        Returns:
            bool: True if deployment successful, False otherwise
        """
        try:
            self.start_time = datetime.now()
            logger.info("Starting deployment process...")
            
            # Install and configure Ngrok
            logger.info("Setting up Ngrok...")
            if not self.ngrok_manager.install_ngrok():
                logger.error("Failed to install Ngrok")
                return False
            
            if not self.ngrok_manager.configure_auth_token():
                logger.error("Failed to configure Ngrok auth token")
                return False
            
            # Start Streamlit
            logger.info("Starting Streamlit application...")
            if not self.streamlit_manager.start_streamlit():
                logger.error("Failed to start Streamlit")
                return False
            
            # Start Ngrok tunnel
            logger.info("Creating Ngrok tunnel...")
            if not self.ngrok_manager.start_ngrok_tunnel(subdomain):
                logger.error("Failed to create Ngrok tunnel")
                self.streamlit_manager.stop_streamlit()
                return False
            
            # Get public URL
            public_url = self.ngrok_manager.get_public_url()
            if not public_url:
                logger.error("Failed to retrieve public URL")
                self.shutdown()
                return False
            
            # Perform health check
            logger.info("Performing health check...")
            if not deployment_health_check(public_url):
                logger.warning("Health check failed, but deployment may still work")
            
            # Log deployment
            create_deployment_log(public_url, self.ngrok_manager.port, self.start_time)
            
            self.is_running = True
            logger.info(f"Deployment successful! Application available at: {public_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            self.shutdown()
            return False
    
    def monitor_deployment(self) -> None:
        """Monitor deployment and restart components if needed."""
        logger.info("Starting deployment monitoring...")
        
        while self.is_running:
            try:
                # Check Streamlit process
                if (self.streamlit_manager.process and 
                    self.streamlit_manager.process.poll() is not None):
                    logger.warning("Streamlit process died, restarting...")
                    if not self.streamlit_manager.start_streamlit():
                        logger.error("Failed to restart Streamlit")
                        break
                
                # Check and restart Ngrok tunnel if needed
                if not self.ngrok_manager.restart_tunnel_if_needed():
                    logger.error("Failed to maintain Ngrok tunnel")
                    break
                
                # Log status
                status = self.ngrok_manager.monitor_tunnel_status()
                if status["is_running"]:
                    logger.info(f"Deployment healthy - URL: {status['public_url']}")
                else:
                    logger.warning("Tunnel not running properly")
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring: {str(e)}")
                time.sleep(10)  # Wait before retry
    
    def shutdown(self) -> None:
        """Shutdown deployment gracefully."""
        logger.info("Shutting down deployment...")
        self.is_running = False
        
        # Stop components
        self.streamlit_manager.stop_streamlit()
        self.ngrok_manager.graceful_shutdown()
        
        logger.info("Deployment shutdown complete")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal")
    global deployment_manager
    if 'deployment_manager' in globals():
        deployment_manager.shutdown()
    sys.exit(0)


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy Heart Disease ML Pipeline with Ngrok"
    )
    parser.add_argument(
        "--app", 
        default="ui/streamlit_app.py",
        help="Path to Streamlit application (default: ui/streamlit_app.py)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Port for Streamlit application (default: 8501)"
    )
    parser.add_argument(
        "--auth-token",
        help="Ngrok authentication token"
    )
    parser.add_argument(
        "--subdomain",
        help="Custom subdomain (requires Ngrok paid plan)"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Skip deployment monitoring"
    )
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create deployment manager
    global deployment_manager
    deployment_manager = DeploymentManager(
        app_path=args.app,
        port=args.port,
        auth_token=args.auth_token
    )
    
    try:
        # Deploy application
        if deployment_manager.deploy(subdomain=args.subdomain):
            print("\n" + "="*60)
            print("🚀 DEPLOYMENT SUCCESSFUL!")
            print("="*60)
            
            public_url = deployment_manager.ngrok_manager.get_public_url()
            if public_url:
                print(f"📱 Public URL: {public_url}")
                print(f"🏠 Local URL:  http://localhost:{args.port}")
                print("="*60)
                print("Press Ctrl+C to stop the deployment")
                print("="*60)
            
            # Start monitoring if requested
            if not args.no_monitor:
                deployment_manager.monitor_deployment()
            else:
                # Just wait for interrupt
                try:
                    while deployment_manager.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        else:
            logger.error("Deployment failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        deployment_manager.shutdown()


if __name__ == "__main__":
    main()