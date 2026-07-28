import os
import streamlit.components.v1 as components

def inject_analytics():
    ga_id = os.environ.get("GA_MEASUREMENT_ID", "")
    if ga_id != "":
        ga_script = f"""
        <script>
            try {{
                const script = window.parent.document.createElement('script');
                script.src = 'https://www.googletagmanager.com/gtag/js?id={ga_id}';
                script.async = true;
                window.parent.document.head.appendChild(script);

                const inlineScript = window.parent.document.createElement('script');
                inlineScript.innerHTML = `
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){{dataLayer.push(arguments);}}
                    gtag('js', new Date());
                    gtag('config', '{ga_id}');
                `;
                window.parent.document.head.appendChild(inlineScript);
            }} catch (e) {{
                console.log("Analytics injection blocked by CORS policy.");
            }}
        </script>
        """
        components.html(ga_script, width=0, height=0)
