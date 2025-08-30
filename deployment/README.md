# Heart Disease ML Pipeline - Deployment Guide

This guide provides comprehensive instructions for deploying the Heart Disease ML Pipeline using Ngrok for public access.

## Overview

The deployment system provides:
- Automated Ngrok installation and configuration
- Streamlit application management
- Public URL generation and monitoring
- Health checking and automatic restart capabilities
- Deployment logging and troubleshooting

## Quick Start

### 1. Basic Deployment

```bash
# Navigate to project root
cd heart-disease-ml-pipeline

# Install deployment dependencies
pip install -r deployment/requirements.txt

# Deploy with default settings
python deployment/deploy.py
```

### 2. Advanced Deployment

```bash
# Deploy with custom port and auth token
python deployment/deploy.py --port 8502 --auth-token YOUR_NGROK_TOKEN

# Deploy with custom subdomain (requires paid Ngrok plan)
python deployment/deploy.py --subdomain my-heart-app --auth-token YOUR_NGROK_TOKEN

# Deploy without monitoring (for CI/CD)
python deployment/deploy.py --no-monitor
```

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Internet connection for Ngrok tunnel
- At least 512MB available RAM
- 100MB free disk space

### Dependencies

Install the required dependencies:

```bash
pip install -r deployment/requirements.txt
```

### Ngrok Account (Optional but Recommended)

1. Sign up for a free Ngrok account at [https://ngrok.com/](https://ngrok.com/)
2. Get your authentication token from the dashboard
3. Use the token with `--auth-token` parameter for better limits

## Deployment Options

### Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--app` | Path to Streamlit app | `ui/streamlit_app.py` | `--app my_app.py` |
| `--port` | Streamlit port | `8501` | `--port 8502` |
| `--auth-token` | Ngrok auth token | None | `--auth-token abc123` |
| `--subdomain` | Custom subdomain | None | `--subdomain my-app` |
| `--no-monitor` | Skip monitoring | False | `--no-monitor` |

### Environment Variables

You can also set configuration via environment variables:

```bash
export NGROK_AUTH_TOKEN="your_token_here"
export STREAMLIT_PORT="8501"
export STREAMLIT_APP_PATH="ui/streamlit_app.py"

python deployment/deploy.py
```

## Deployment Process

### Step-by-Step Process

1. **Ngrok Installation**: Automatically downloads and installs Ngrok if not present
2. **Authentication**: Configures Ngrok with your auth token (if provided)
3. **Streamlit Configuration**: Optimizes Streamlit settings for public access
4. **Application Start**: Launches the Streamlit application
5. **Tunnel Creation**: Establishes Ngrok tunnel to your local application
6. **Health Check**: Verifies the application is accessible via public URL
7. **Monitoring**: Continuously monitors and restarts components if needed

### What Happens During Deployment

```
🔧 Installing Ngrok...
🔑 Configuring authentication...
🚀 Starting Streamlit application...
🌐 Creating Ngrok tunnel...
🏥 Performing health check...
📝 Logging deployment...
✅ Deployment successful!

📱 Public URL: https://abc123.ngrok.io
🏠 Local URL:  http://localhost:8501
```

## Monitoring and Management

### Automatic Monitoring

The deployment system includes automatic monitoring that:
- Checks Streamlit process health every 30 seconds
- Monitors Ngrok tunnel status
- Automatically restarts failed components
- Logs all activities and errors

### Manual Management

#### Check Deployment Status

```bash
# View deployment logs
cat deployment/deployment_log.json

# Check running processes
ps aux | grep streamlit
ps aux | grep ngrok
```

#### Restart Components

```bash
# Kill existing processes
pkill -f streamlit
pkill -f ngrok

# Restart deployment
python deployment/deploy.py
```

### Deployment Logs

Deployment activities are logged to `deployment/deployment_log.json`:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "public_url": "https://abc123.ngrok.io",
  "port": 8501,
  "platform": "Windows",
  "status": "active"
}
```

## Troubleshooting

### Common Issues

#### 1. Ngrok Installation Fails

**Problem**: Automatic Ngrok installation fails

**Solutions**:
```bash
# Manual installation on Windows
curl -o ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip
unzip ngrok.zip

# Manual installation on macOS
brew install ngrok

# Manual installation on Linux
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
unzip ngrok-v3-stable-linux-amd64.zip
chmod +x ngrok
```

#### 2. Port Already in Use

**Problem**: `Port 8501 is already in use`

**Solutions**:
```bash
# Use different port
python deployment/deploy.py --port 8502

# Kill existing process
lsof -ti:8501 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8501   # Windows (find PID and kill)
```

#### 3. Tunnel Connection Fails

**Problem**: Cannot establish Ngrok tunnel

**Solutions**:
```bash
# Check internet connection
ping ngrok.com

# Try with auth token
python deployment/deploy.py --auth-token YOUR_TOKEN

# Check firewall settings
# Ensure ports 4040 and your app port are not blocked
```

#### 4. Application Not Accessible

**Problem**: Public URL returns error or timeout

**Solutions**:
```bash
# Check local application first
curl http://localhost:8501

# Verify Streamlit is running
ps aux | grep streamlit

# Check Ngrok tunnel status
curl http://localhost:4040/api/tunnels

# Restart deployment
python deployment/deploy.py
```

#### 5. Authentication Issues

**Problem**: Ngrok authentication fails

**Solutions**:
```bash
# Verify token format (should be long alphanumeric string)
ngrok config add-authtoken YOUR_TOKEN

# Check account status at ngrok.com
# Ensure token is active and not expired
```

### Advanced Troubleshooting

#### Debug Mode

Enable verbose logging:

```bash
# Set debug logging
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
exec(open('deployment/deploy.py').read())
"
```

#### Network Diagnostics

```bash
# Check local connectivity
curl -I http://localhost:8501

# Check Ngrok API
curl http://localhost:4040/api/tunnels

# Test public URL
curl -I https://your-ngrok-url.ngrok.io
```

#### Process Monitoring

```bash
# Monitor resource usage
top -p $(pgrep -f streamlit)
top -p $(pgrep -f ngrok)

# Check system resources
df -h  # Disk space
free -h  # Memory usage
```

## Security Considerations

### Public Access Security

- The application will be publicly accessible via Ngrok URL
- Consider implementing authentication in your Streamlit app
- Monitor access logs for suspicious activity
- Use HTTPS URLs when available

### Data Privacy

- Ensure no sensitive data is exposed in the application
- Consider data anonymization for public deployments
- Review Streamlit app for any debug information exposure

### Ngrok Security

- Keep your auth token secure and private
- Regularly rotate auth tokens
- Monitor your Ngrok dashboard for unauthorized usage
- Use custom subdomains for better URL control (paid feature)

## Performance Optimization

### Streamlit Optimization

```python
# Add to your Streamlit app for better performance
import streamlit as st

# Cache expensive operations
@st.cache_data
def load_model():
    # Your model loading code
    pass

# Optimize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
```

### Ngrok Optimization

- Use paid Ngrok plan for better performance and custom domains
- Consider regional tunnels for better latency
- Monitor bandwidth usage in Ngrok dashboard

## Alternative Deployment Options

### Cloud Deployment

For production use, consider these alternatives:

1. **Streamlit Cloud**: Native Streamlit hosting
2. **Heroku**: Easy cloud deployment
3. **AWS/GCP/Azure**: Full cloud infrastructure
4. **Docker**: Containerized deployment

### Local Network Deployment

```bash
# Deploy on local network only
streamlit run ui/streamlit_app.py --server.address 0.0.0.0
```

## Support and Resources

### Documentation

- [Ngrok Documentation](https://ngrok.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Project GitHub Repository](https://github.com/your-repo)

### Getting Help

1. Check this troubleshooting guide
2. Review deployment logs
3. Search existing GitHub issues
4. Create new issue with detailed error information

### Contributing

To contribute to the deployment system:

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request with detailed description

## Changelog

### Version 1.0.0
- Initial deployment system
- Automatic Ngrok installation
- Streamlit configuration management
- Health checking and monitoring
- Comprehensive error handling

---

For additional support, please refer to the project documentation or create an issue in the GitHub repository.