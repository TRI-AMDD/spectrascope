import os
import pathlib

import streamlit as st
from dotenv import load_dotenv
from streamlit_cognito_auth import CognitoHostedUIAuthenticator

load_dotenv(pathlib.Path(__file__).parent / ".env")


def authenticate_user():
    pool_id = os.getenv("COGNITO_POOL_ID")
    app_client_id = os.getenv("COGNITO_APP_CLIENT_ID")
    app_client_secret = os.getenv("COGNITO_APP_CLIENT_SECRET")
    cognito_domain = os.getenv("COGNITO_DOMAIN")
    redirect_uri = os.getenv("COGNITO_REDIRECT_URI")

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
