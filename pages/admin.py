import streamlit as st
import pandas as pd
import db

# Need to check authentication status directly from session_state
if "authentication_status" not in st.session_state or not st.session_state["authentication_status"]:
    st.error("Access Denied. You must be logged in to view this page.")
    st.stop()

# Basic authorization check: Ensure the user is an admin
if st.session_state.get("username") != "admin":
    st.error("Access Denied. You do not have administrator privileges.")
    st.stop()

st.title("🛡️ Admin Dashboard: Audit Logs")
st.write("Review all user interactions within the Data Room.")

logs = db.get_all_logs()

if not logs:
    st.info("No logs found.")
else:
    df = pd.DataFrame(logs)
    
    # Simple metric cards
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", len(df[df['action'] == 'QUERY']))
    with col2:
        st.metric("Total PDF Downloads", len(df[df['action'] == 'DOWNLOAD_PDF']))
        
    st.subheader("Detailed Log Data")
    # Display the dataframe with Streamlit's interactive table
    st.dataframe(
        df,
        column_config={
            "timestamp": "Timestamp",
            "user_email": "Investor Email",
            "action": "Action",
            "question": "Question Asked",
            "answer": "AI Response"
        },
        use_container_width=True,
        hide_index=True
    )
