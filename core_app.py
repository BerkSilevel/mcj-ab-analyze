import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tempfile import NamedTemporaryFile
import streamlit as st
from google.cloud import bigquery
import plotly.graph_objects as go

def run_retention_dashboard():
    # --- Service Account Key: st.secrets üzerinden geçici dosya oluştur ---
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
        with NamedTemporaryFile(delete=False, mode='w', suffix=".json") as tmp:
            tmp.write(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            tmp.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

    client = bigquery.Client()
    query = st.secrets["query"]

    # --- Veri Yükleme: Session'dan ya da BigQuery'den ---
    if st.button("Pull the Dataset From BigQuery 📡") or "retention_df" not in st.session_state:
        try:
            df = client.query(query).result().to_dataframe()
            st.session_state["retention_df"] = df
            st.success("✅ Veri başarıyla çekildi.")
        except Exception as e:
            st.error(f"❌ Hata oluştu: {e}")
            return

    df = st.session_state.get("retention_df")

    if df is None or df.empty:
        st.info("📌 Lütfen önce 'Veriyi Çek 📡' butonuna tıklayarak veriyi yükleyin.")
        return

    # --- Kolon İsimlerini Düzenle ---
    df.rename(columns={
        'D1_user_count': 'retention_day_1',
        'D3_user_count': 'retention_day_3'
    }, inplace=True)

    # --- Dimension Seçimi ---
    st.subheader("🔍 Choose Dimension ")
    dimension_options = ["Experiment_ID", "Experiment_Variant", "OS"]
    selected_dimensions = {}

    for dim in dimension_options:
        if dim in df.columns:
            if st.checkbox(f"{dim}", value=False):
                values = df[dim].dropna().unique().tolist()
                selected = st.multiselect(f"{dim} için değer(ler)", values, default=values)
                if selected:
                    selected_dimensions[dim] = selected

    if selected_dimensions:
        filtered_df = df.copy()
        for col, vals in selected_dimensions.items():
            filtered_df = filtered_df[filtered_df[col].isin(vals)]

        groupby_dims = list(selected_dimensions.keys())
        num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        agg_df = filtered_df.groupby(groupby_dims)[num_cols].sum().reset_index()

        if 'Cost' in agg_df.columns and 'Installs' in agg_df.columns:
            agg_df['CPI'] = agg_df['Cost'] / agg_df['Installs'].replace(0, np.nan)

        if 'Ad_Revenue' in agg_df.columns:
            agg_df['ARPU'] = agg_df['Ad_Revenue'] / agg_df['Installs'].replace(0, np.nan)

        try:
            if {'retention_day_1', 'retention_day_3', 'Installs'}.issubset(agg_df.columns):
                agg_df['D1_retention'] = agg_df['retention_day_1'] / agg_df['Installs']
                agg_df['D3_retention'] = agg_df['retention_day_3'] / agg_df['Installs']

                model_path = Path(__file__).parent / "model_d4_d15.pkl"
                model = joblib.load(model_path)

                model_input = agg_df[['D1_retention', 'D3_retention']]
                preds = model.predict(model_input)

                pred_cols = [f'Predicted_D{i}_Retention' for i in range(4, 16)]
                pred_df = pd.DataFrame(preds, columns=pred_cols)
                agg_df = pd.concat([agg_df.reset_index(drop=True), pred_df], axis=1)

                # D1-D3 retention tahmini
                agg_df['Predicted_D1_Retention'] = agg_df['D1_retention']
                agg_df['Predicted_D2_Retention'] = agg_df['D1_retention'] * ((agg_df['D3_retention'] / agg_df['D1_retention']) ** (1 / 2))
                agg_df['Predicted_D3_Retention'] = agg_df['D3_retention']

                all_pred_cols = [f'Predicted_D{i}_Retention' for i in range(1, 16)]
                days = list(range(1, 16))

                st.subheader("📊 Aggregated + Tahminli Veri")
                st.dataframe(agg_df)

                st.subheader("📈 Graph of D1–D15 Retention")
                fig = go.Figure()

                for i, row in agg_df.iterrows():
                    y_vals = row[all_pred_cols].values
                    label = " - ".join([str(row.get(col, "")) for col in groupby_dims])
                    fig.add_trace(go.Scatter(
                        x=days,
                        y=y_vals,
                        mode="lines+markers",
                        name=label,
                        hovertemplate="Gün %{x}<br>Retention: %{y:.1%}<extra></extra>"
                    ))

                fig.update_layout(
                    xaxis_title="Day",
                    yaxis_title="Estimated Retention Rate",
                    yaxis_tickformat=".0%",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("💰 Estimated ARPU & LTV ")

                if {'ARPU'}.issubset(agg_df.columns):
                    ltv_values = []
                    for i, row in agg_df.iterrows():
                        retention_curve = np.array([row[f'Predicted_D{d}_Retention'] for d in range(1, 16)])
                        ltv = row['ARPU'] * np.sum(retention_curve)
                        ltv_values.append(ltv)

                    agg_df['Predicted_LTV'] = ltv_values

                    st.dataframe(agg_df[groupby_dims + ['ARPU']])

                    fig_ltv = go.Figure()
                    for i, row in agg_df.iterrows():
                        ltv_curve = row['ARPU'] * np.cumsum([row[f'Predicted_D{d}_Retention'] for d in range(1, 16)])
                        fig_ltv.add_trace(go.Scatter(
                            x=list(range(1, 16)),
                            y=ltv_curve,
                            mode="lines+markers",
                            name=" - ".join([str(row.get(col, "")) for col in groupby_dims]),
                            hovertemplate="Gün %{x}<br>LTV: $%{y:.2f}<extra></extra>"
                        ))

                    fig_ltv.update_layout(
                        title="📈 Cumulative LTV",
                        xaxis_title="Day ",
                        yaxis_title="Cumulative LTV ($)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_ltv, use_container_width=True)
                else:
                    st.warning("ARPU hesaplaması için gerekli veriler eksik.")
            else:
                st.warning("D1, D3 veya Installs kolonları eksik.")
        except Exception as e:
            st.error(f"❌ Tahmin sırasında hata oluştu: {e}")
    else:
        st.info("Lütfen en az bir dimension seçin.")
