#!/usr/bin/env python3
"""
Test script for deployment system validation.

This script tests the core functionality of the Ngrok deployment system
without actually starting the services.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deployment.ngrok_setup import NgrokManager, create_deployment_log, deployment_health_check
from deployment.deploy import StreamlitManager, DeploymentManager


def test_ngrok_manager_initialization():
    """Test NgrokManager initialization."""
    print("Testing NgrokManager initialization...")
    
    manager = NgrokManager(port=8501, auth_token="test_token")
    assert manager.port == 8501
    assert manager.auth_token == "test_token"
    assert manager.tunnel_process is None
    assert manager.public_url is None
    
    print("✅ NgrokManager initialization test passed")


def test_streamlit_manager_initialization():
    """Test StreamlitManager initialization."""
    print("Testing StreamlitManager initialization...")
    
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
        tmp.write(b"# Test Streamlit app")
        tmp_path = tmp.name
    
    try:
        manager = StreamlitManager(tmp_path, port=8502)
        assert manager.port == 8502
        assert manager.app_path == Path(tmp_path)
        assert manager.process is None
        
        print("✅ StreamlitManager initialization test passed")
    finally:
        Path(tmp_path).unlink()


def test_deployment_manager_initialization():
    """Test DeploymentManager initialization."""
    print("Testing DeploymentManager initialization...")
    
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
        tmp.write(b"# Test Streamlit app")
        tmp_path = tmp.name
    
    try:
        manager = DeploymentManager(tmp_path, port=8503, auth_token="test_token")
        assert manager.streamlit_manager.port == 8503
        assert manager.ngrok_manager.port == 8503
        assert manager.ngrok_manager.auth_token == "test_token"
        assert not manager.is_running
        
        print("✅ DeploymentManager initialization test passed")
    finally:
        Path(tmp_path).unlink()


def test_deployment_log_creation():
    """Test deployment log creation."""
    print("Testing deployment log creation...")
    
    from datetime import datetime
    import json
    import tempfile
    import os
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_dir)
        
        try:
            # Create deployment directory
            Path("deployment").mkdir(exist_ok=True)
            
            # Test log creation
            test_url = "https://test123.ngrok.io"
            test_port = 8501
            test_time = datetime.now()
            
            create_deployment_log(test_url, test_port, test_time)
            
            # Verify log file was created
            log_file = Path("deployment/deployment_log.json")
            assert log_file.exists()
            
            # Verify log content
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            assert len(logs) == 1
            assert logs[0]["public_url"] == test_url
            assert logs[0]["port"] == test_port
            assert logs[0]["status"] == "active"
            
            print("✅ Deployment log creation test passed")
            
        finally:
            os.chdir(original_cwd)


def test_ngrok_path_detection():
    """Test Ngrok path detection for different platforms."""
    print("Testing Ngrok path detection...")
    
    manager = NgrokManager()
    ngrok_path = manager._get_ngrok_path()
    
    # Should return a Path object
    assert isinstance(ngrok_path, Path)
    
    # Should have appropriate extension based on platform
    import platform
    if platform.system() == "Windows":
        assert ngrok_path.name == "ngrok.exe"
    else:
        assert ngrok_path.name == "ngrok"
    
    print("✅ Ngrok path detection test passed")


def test_streamlit_config_generation():
    """Test Streamlit configuration generation."""
    print("Testing Streamlit configuration generation...")
    
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
        tmp.write(b"# Test Streamlit app")
        tmp_path = tmp.name
    
    try:
        manager = StreamlitManager(tmp_path, port=8504)
        
        # Mock the config directory to avoid modifying user's actual config
        with patch('pathlib.Path.home') as mock_home:
            with tempfile.TemporaryDirectory() as tmp_dir:
                mock_home.return_value = Path(tmp_dir)
                
                # Test configuration
                manager.configure_streamlit_for_ngrok()
                
                # Check if config file was created
                config_file = Path(tmp_dir) / ".streamlit" / "config.toml"
                assert config_file.exists()
                
                # Check config content
                config_content = config_file.read_text()
                assert f"port = {manager.port}" in config_content
                assert "enableCORS = false" in config_content
                
        print("✅ Streamlit configuration test passed")
        
    finally:
        Path(tmp_path).unlink()


def test_health_check_mock():
    """Test health check functionality with mocked requests."""
    print("Testing health check functionality...")
    
    # Mock successful health check
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = deployment_health_check("https://test.ngrok.io", max_attempts=1)
        assert result is True
        
    # Mock failed health check
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = deployment_health_check("https://test.ngrok.io", max_attempts=1)
        assert result is False
    
    print("✅ Health check test passed")


def run_all_tests():
    """Run all deployment system tests."""
    print("🧪 Running Deployment System Tests")
    print("=" * 50)
    
    try:
        test_ngrok_manager_initialization()
        test_streamlit_manager_initialization()
        test_deployment_manager_initialization()
        test_deployment_log_creation()
        test_ngrok_path_detection()
        test_streamlit_config_generation()
        test_health_check_mock()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("✅ Deployment system is ready for use")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)