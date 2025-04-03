import psutil
import time
import numpy as np
import joblib
import threading
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest


# Collect system data for training
def collect_training_data():
    process_data = []
    for _ in range(10):  # Reduced iterations for faster startup
        snapshot = []
        for p in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                snapshot.append((p.pid, p.name(), p.cpu_percent(), p.memory_percent()))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue  # Skip processes that no longer exist or are inaccessible
        process_data.extend(snapshot)
        time.sleep(0.2)  # Reduced sleep time for faster training

    if process_data:
        process_array = np.array(process_data)[:, 2:].astype(float)
        clf = IsolationForest(contamination=0.05, random_state=42)
        clf.fit(process_array)
        joblib.dump(clf, 'anomaly_model.pkl')  # Save the model


# Function to terminate a process
def kill_process(pid):
    try:
        p = psutil.Process(pid)
        p.terminate()
        st.success(f"Process {pid} has been terminated.")
    except Exception as e:
        st.error(f"Could not terminate process {pid}: {e}")


# Function to monitor processes and detect anomalies
def monitor_system():
    try:
        clf = joblib.load('anomaly_model.pkl')  # Load trained model
    except FileNotFoundError:
        st.error("Error: Model file not found. Run collect_training_data() first.")
        return

    process_snapshot = []
    for p in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            process_snapshot.append((p.pid, p.name(), p.cpu_percent(), p.memory_percent()))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue  # Skip processes that no longer exist or are inaccessible

    if process_snapshot:
        process_snapshot.sort(key=lambda x: x[2], reverse=True)  # Sort by CPU usage
        process_array = np.array([p[2:] for p in process_snapshot], dtype=float)
        predictions = clf.predict(process_array)

        df = pd.DataFrame(process_snapshot, columns=["PID", "Process Name", "CPU Usage", "Memory Usage"])
        df["Status"] = ["Normal" if predictions[i] == 1 else "Anomaly ⚠️" for i in range(len(predictions))]
        return df
    return None


# Streamlit UI
def main():
    st.title("AI Task Manager")
    st.write("### Monitor and manage system processes in real-time with AI-driven anomaly detection.")

    if st.button("Train AI Model (First Time Only)"):
        with st.spinner("Training model, please wait..."):
            collect_training_data()
            st.success("Model training completed!")

    if st.button("Refresh Process List"):
        df = monitor_system()
        if df is not None:
            st.dataframe(df)
        else:
            st.warning("No process data available.")

    pid_to_kill = st.number_input("Enter PID to Kill", min_value=1, step=1)
    if st.button("Kill Process"):
        kill_process(pid_to_kill)


if __name__ == "__main__":
    main()