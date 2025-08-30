"""
Ngrok Setup and Management Module

This module provides automated Ngrok tunnel management for deploying
the Heart Disease ML Pipeline Streamlit application with public access.
"""

import os
import sys
import json
import time
import requests
import subprocess
import platform
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NgrokManager:
    """Manages Ngrok tunnel creation, monitoring, and cleanup."""
    
    def __init__(self, port: int = 8501, auth_token: Optional[str] = None):
        """
        Initialize NgrokManager.
        
        Args:
            port: Port number for Streamlit application (default: 8501)
            auth_token: Ngrok authentication token (optional)
        """
        self.port = port
        self.auth_token = auth_token
        self.tunnel_process = None
        self.public_url = None
        self.ngrok_path = self._get_ngrok_path()
        
    def _get_ngrok_path(self) -> Path:
        """Get the expected path for ngrok executable."""
        if platform.system() == "Windows":
            return Path("ngrok.exe")
        else:
            return Path("ngrok")
    
    def install_ngrok(self) -> bool:
        """
        Automatically install Ngrok if not present.
        
        Returns:
            bool: True if installation successful, False otherwise
        """
        try:
            # Check if ngrok is already installed
            if self._is_ngrok_installed():
                logger.info("Ngrok is already installed")
                return True
            
            logger.info("Installing Ngrok...")
            
            # Determine download URL based on platform
            system = platform.system().lower()
            arch = platform.machine().lower()
            
            if system == "windows":
                if "64" in arch or "amd64" in arch:
                    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
                else:
                    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-386.zip"
            elif system == "darwin":  # macOS
                if "arm" in arch or "m1" in arch.lower():
                    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-arm64.zip"
                else:
                    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip"
            else:  # Linux
                if "64" in arch or "amd64" in arch:
                    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip"
                else:
                    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-386.zip"
            
            # Download ngrok
            logger.info(f"Downloading Ngrok from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save and extract
            zip_path = Path("ngrok.zip")
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract ngrok
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up zip file
            zip_path.unlink()
            
            # Make executable on Unix systems
            if system != "windows":
                os.chmod(self.ngrok_path, 0o755)
            
            logger.info("Ngrok installation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install Ngrok: {str(e)}")
            return False
    
    def _is_ngrok_installed(self) -> bool:
        """Check if ngrok is installed and accessible."""
        try:
            # Check if ngrok executable exists locally
            if self.ngrok_path.exists():
                return True
            
            # Check if ngrok is in PATH
            result = subprocess.run(
                ["ngrok", "version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def configure_auth_token(self) -> bool:
        """
        Configure Ngrok authentication token.
        
        Returns:
            bool: True if configuration successful, False otherwise
        """
        if not self.auth_token:
            logger.warning("No auth token provided. Using free tier limitations.")
            return True
        
        try:
            cmd = [str(self.ngrok_path), "config", "add-authtoken", self.auth_token]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("Ngrok auth token configured successfully")
                return True
            else:
                logger.error(f"Failed to configure auth token: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring auth token: {str(e)}")
            return False
    
    def start_ngrok_tunnel(self, subdomain: Optional[str] = None) -> bool:
        """
        Start Ngrok tunnel for the specified port.
        
        Args:
            subdomain: Custom subdomain (requires paid plan)
            
        Returns:
            bool: True if tunnel started successfully, False otherwise
        """
        try:
            # Build ngrok command
            cmd = [str(self.ngrok_path), "http", str(self.port), "--log=stdout"]
            
            if subdomain:
                cmd.extend(["--subdomain", subdomain])
            
            # Start ngrok process
            logger.info(f"Starting Ngrok tunnel on port {self.port}")
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for tunnel to establish and get public URL
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    url = self.get_public_url()
                    if url:
                        self.public_url = url
                        logger.info(f"Ngrok tunnel established: {url}")
                        return True
                except Exception:
                    pass
                
                time.sleep(1)
            
            logger.error("Failed to establish Ngrok tunnel within timeout")
            self.stop_tunnel()
            return False
            
        except Exception as e:
            logger.error(f"Error starting Ngrok tunnel: {str(e)}")
            return False
    
    def get_public_url(self) -> Optional[str]:
        """
        Retrieve the public URL from Ngrok API.
        
        Returns:
            str: Public URL if available, None otherwise
        """
        try:
            # Query Ngrok local API
            response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            tunnels = data.get("tunnels", [])
            
            for tunnel in tunnels:
                if tunnel.get("proto") == "https":
                    return tunnel.get("public_url")
            
            # Fallback to http if https not available
            for tunnel in tunnels:
                if tunnel.get("proto") == "http":
                    return tunnel.get("public_url")
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not retrieve public URL: {str(e)}")
            return None
    
    def monitor_tunnel_status(self) -> Dict[str, Any]:
        """
        Monitor tunnel status and health.
        
        Returns:
            dict: Status information including connection state and metrics
        """
        status = {
            "is_running": False,
            "public_url": None,
            "connections": 0,
            "bytes_in": 0,
            "bytes_out": 0,
            "last_check": datetime.now().isoformat()
        }
        
        try:
            if self.tunnel_process and self.tunnel_process.poll() is None:
                status["is_running"] = True
                
                # Get tunnel information
                response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    tunnels = data.get("tunnels", [])
                    
                    if tunnels:
                        tunnel = tunnels[0]
                        status["public_url"] = tunnel.get("public_url")
                        
                        # Get metrics if available
                        metrics = tunnel.get("metrics", {})
                        status["connections"] = metrics.get("conns", {}).get("count", 0)
                        status["bytes_in"] = metrics.get("http", {}).get("bytes_in", 0)
                        status["bytes_out"] = metrics.get("http", {}).get("bytes_out", 0)
            
        except Exception as e:
            logger.debug(f"Error monitoring tunnel status: {str(e)}")
        
        return status
    
    def restart_tunnel_if_needed(self) -> bool:
        """
        Restart tunnel if it's not running properly.
        
        Returns:
            bool: True if restart successful or not needed, False otherwise
        """
        status = self.monitor_tunnel_status()
        
        if not status["is_running"]:
            logger.info("Tunnel not running, attempting restart...")
            self.stop_tunnel()
            return self.start_ngrok_tunnel()
        
        return True
    
    def stop_tunnel(self) -> None:
        """Stop the Ngrok tunnel and cleanup."""
        try:
            if self.tunnel_process:
                logger.info("Stopping Ngrok tunnel...")
                self.tunnel_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.tunnel_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Forcing Ngrok process termination...")
                    self.tunnel_process.kill()
                    self.tunnel_process.wait()
                
                self.tunnel_process = None
                self.public_url = None
                logger.info("Ngrok tunnel stopped")
                
        except Exception as e:
            logger.error(f"Error stopping tunnel: {str(e)}")
    
    def graceful_shutdown(self) -> None:
        """Perform graceful shutdown with cleanup."""
        logger.info("Performing graceful shutdown...")
        self.stop_tunnel()
        
        # Additional cleanup if needed
        try:
            # Kill any remaining ngrok processes
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/f", "/im", "ngrok.exe"], 
                             capture_output=True, timeout=10)
            else:
                subprocess.run(["pkill", "-f", "ngrok"], 
                             capture_output=True, timeout=10)
        except Exception:
            pass  # Ignore errors in cleanup
        
        logger.info("Graceful shutdown completed")


def create_deployment_log(public_url: str, port: int, start_time: datetime) -> None:
    """
    Create deployment log entry.
    
    Args:
        public_url: The public URL of the deployment
        port: Port number used
        start_time: Deployment start time
    """
    log_entry = {
        "timestamp": start_time.isoformat(),
        "public_url": public_url,
        "port": port,
        "platform": platform.system(),
        "python_version": sys.version,
        "status": "active"
    }
    
    log_file = Path("deployment/deployment_log.json")
    
    # Load existing logs
    logs = []
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    
    # Add new log entry
    logs.append(log_entry)
    
    # Keep only last 50 entries
    logs = logs[-50:]
    
    # Save updated logs
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    logger.info(f"Deployment logged: {public_url}")


def deployment_health_check(url: str, max_attempts: int = 5) -> bool:
    """
    Verify application accessibility through the public URL.
    
    Args:
        url: Public URL to check
        max_attempts: Maximum number of health check attempts
        
    Returns:
        bool: True if application is accessible, False otherwise
    """
    logger.info(f"Performing health check on {url}")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.info("Health check passed - application is accessible")
                return True
            else:
                logger.warning(f"Health check attempt {attempt + 1}: HTTP {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Health check attempt {attempt + 1} failed: {str(e)}")
        
        if attempt < max_attempts - 1:
            time.sleep(5)  # Wait before retry
    
    logger.error("Health check failed - application may not be accessible")
    return False


if __name__ == "__main__":
    # Example usage
    manager = NgrokManager(port=8501)
    
    try:
        # Install and configure ngrok
        if manager.install_ngrok():
            manager.configure_auth_token()
            
            # Start tunnel
            if manager.start_ngrok_tunnel():
                url = manager.get_public_url()
                if url:
                    print(f"Application available at: {url}")
                    create_deployment_log(url, 8501, datetime.now())
                    
                    # Keep tunnel running
                    try:
                        while True:
                            time.sleep(30)
                            if not manager.restart_tunnel_if_needed():
                                break
                    except KeyboardInterrupt:
                        print("\nShutting down...")
                        
    finally:
        manager.graceful_shutdown()