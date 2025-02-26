import os
import pathlib

from dotenv import load_dotenv
import streamlit as st

load_dotenv(pathlib.Path(__file__).parent / ".env")


def inject_google_analytics():
    ga_tracking_id = os.getenv("GA_TRACKING_ID", "G-TEST1234")
    ga_script = f"""
    <!-- Google Analytics (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={ga_tracking_id}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag() {{ dataLayer.push(arguments); }}
      gtag('js', new Date());
      gtag('config', '{ga_tracking_id}');
    </script>
    """

    st.markdown(ga_script, unsafe_allow_html=True)
