import os

c = get_config()

# JupyterHub configuration
c.JupyterHub.ip = '0.0.0.0'
c.JupyterHub.port = 8000

# Use PAM Authenticator
c.JupyterHub.authenticator_class = 'jupyterhub.auth.PAMAuthenticator'
c.Authenticator.admin_users = {'adi'}

# Allow any user to access
c.Authenticator.allow_all = True
c.PAMAuthenticator.open_sessions = False

# Note: Password is set in the Dockerfile with: RUN echo "adi:password" | chpasswd

c.JupyterHub.spawner_class = 'simple'
c.Spawner.default_url = '/lab'