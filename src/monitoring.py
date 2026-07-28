import os
import streamlit.components.v1 as components

def inject_analytics():
    ga_id = os.environ.get("GA_MEASUREMENT_ID", "")
    if ga_id != "":
        ga_script = f"""
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id={ga_id}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{ga_id}');
        </script>
        """
        components.html(ga_script, width=0, height=0)
