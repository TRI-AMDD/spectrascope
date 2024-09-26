import os

import streamlit as st
from streamlit_cognito_auth import CognitoHostedUIAuthenticator


def authenticate_user():
    pool_id = os.getenv("COGNITO_POOL_ID", "us-east-1_bLHpPO2vu")
    app_client_id = os.getenv("COGNITO_APP_CLIENT_ID", "5718oir6vu2lf675co2cs36bu0")
    app_client_secret = os.getenv(
        "COGNITO_APP_CLIENT_SECRET",
        "1ktad0valnammk2q9m8ep4jflcv5umu7q1kvf948813q38k9qj0k",
    )
    cognito_domain = os.getenv(
        "COGNITO_DOMAIN", "https://em-stage-apps.auth.us-east-1.amazoncognito.com"
    )
    redirect_uri = os.getenv("COGNITO_REDIRECT_URI", "http://localhost:8501")

    authenticator = CognitoHostedUIAuthenticator(
        pool_id=pool_id,
        app_client_id=app_client_id,
        app_client_secret=app_client_secret,
        cognito_domain=cognito_domain,
        redirect_uri=redirect_uri,
        use_cookies=False,
    )

    is_logged_in = authenticator.login()
    if not is_logged_in:
        st.warning("Please login to continue")
        st.stop()

    return authenticator


def render_sidebar(authenticator):
    def logout():
        authenticator.logout()

    with st.sidebar:
        st.text(f"Welcome,\n{authenticator.get_username()}")
        st.text(f"Your email is:\n{authenticator.get_email()}")
        st.button("Logout", "logout_btn", on_click=logout)
