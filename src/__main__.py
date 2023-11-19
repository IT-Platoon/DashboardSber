import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import streamlit as st

from graphics import load_file, plot_increasing, plot_box, plot_charTS


st.set_page_config(page_title="Dashboard Sber")
st.title('Dashboard Sber')

tab1, tab2, tab3 = st.tabs(["Dashboard", "Prediction", "Profiles"])

with tab1:
   st.title("Построение дашбордов")
   uploaded_file_acquiring = st.file_uploader(
      "Выберите файл эквайринга",
      key="dashboard_acquiring",
   )
   df_acquiring, df_general = None, None
   if (
      uploaded_file_acquiring and uploaded_file_acquiring.name.endswith(".xlsb")
   ):
      my_bar = st.progress(0, text="Обработка файла эквайринга")
      df_acquiring = load_file(file_path=uploaded_file_acquiring.getvalue(), header=1, need_data_fix=True)
      my_bar.progress(50, text="Обработка файла с основной информацией")
      df_general = load_file(file_path='./src/dataset/general.xlsb', header=0)
      my_bar.progress(100, text="Обработка файлов завершена")
      my_bar.empty()
   

   if df_acquiring is not None and df_general is not None:

      tmp_cols = ['клиент', 'Сегмент id', 'Кластер', 'Средний возраст работников', 'Тип организации']
      df_acquiring_merged = pd.merge(df_general[tmp_cols], df_acquiring, on='клиент', how='outer')
      series_cols = df_acquiring.columns[df_acquiring.columns.str.contains("клиент") == False].values
      level_cols = df_acquiring.columns[df_acquiring.columns.str.contains("клиент")].values
      data = pd.DataFrame(series_cols, columns=["ds"])
      data['ds'] = pd.to_datetime(data['ds'], format='%Y-%m-%d')
      data.loc[:, "y"] = df_acquiring[series_cols].sum().values
      data = data.set_index('ds')

      for col in tmp_cols:

         fig = plot_increasing(
            st,
            df=df_acquiring_merged,
            feature=col,
            series=series_cols,
            name='продукта',
         )
         st.pyplot(fig)


      fig = plot_box(st, data)
      st.pyplot(fig)

      fig = plot_charTS(st, data["y"])
      st.pyplot(fig)

with tab2:
   st.title("Предсказание значений")
   date = st.date_input(
      "Введите дату для предсказания",
      value="today",
      format="YYYY-MM-DD",
      label_visibility="visible",
   )
   

with tab3:
   st.title("Профили решений")

   uploaded_file_general = st.file_uploader(
      "Выберите файл с основной информацией",
      key="profile_general",
   )
   df_general = None
   if (
      uploaded_file_general and uploaded_file_general.name.endswith(".xlsb")
   ):
      my_bar = st.progress(0, text="Обработка главного файла")
      df_general = load_file(file_path=uploaded_file_general.getvalue(), header=0)
      my_bar.progress(100, text="Обработка файлов")
      my_bar.empty()
   

   if df_general is not None:
      cat_columns = df_general.select_dtypes(exclude=['int64', 'float64']).columns[1:]

      coders = {}
      def use_encoder(seria: pd.Series, name: str):
         seria = seria.astype('str')
         le = LabelEncoder()
         coders[name] = le
         encode_seria = le.fit_transform(seria)
         return encode_seria
      reg_general = df_general.copy().fillna(
      value={'Тип организации': df_general[['Тип организации']].mode()['Тип организации'][0]}
      )
      for column in reg_general.columns:
         if column != 'клиент':
            new_column = use_encoder(reg_general[column], column)
            reg_general[column] = new_column
      
      min_corr = 0.3
      max_corr = 0.75
      self_corr = 1.0
      target_column = "Кластер"
      correlation_general = reg_general.iloc[:, 1:].corr()

      correlation = correlation_general
      correlation = (abs(correlation) >= min_corr) & (abs(correlation) < max_corr) & (correlation != self_corr)
      useful_features = []
      for column in correlation.columns:
         if correlation[column].any(axis=0) == False:
            useful_features.append(column)
      
      drop_train = correlation_general.drop(labels=useful_features, axis=0)
      drop_train = drop_train[list(drop_train.index)]
      correlation = drop_train

      corr_columns = []
      for first_column in correlation.columns[:-1]:
         for second_column in correlation.columns[1:]:
            if drop_train[first_column][second_column] > max_corr and drop_train[first_column][second_column] != self_corr:
                  corr_columns.append([first_column, second_column])
      corr_columns_values = {}
      for correlated in corr_columns:
         if correlated[0] not in corr_columns_values.keys():
            corr_columns_values[correlated[0]] = 1
         else:
            corr_columns_values[correlated[0]] += 1
      expelled_columns = []
      value_max = 0
      for feature in corr_columns_values.keys():
         if corr_columns_values[feature] > value_max:
            expelled_columns.clear()
            expelled_columns.append(feature)
            value_max = corr_columns_values[feature]
         elif corr_columns_values[feature] == value_max:
            expelled_columns.append(feature)
         else:
            pass
      if value_max != 1:
         correlation = correlation.drop(labels=expelled_columns, axis=0)
         correlation = correlation[list(correlation.index)]

      correct = df_general.copy()
      correct = correct[correlation.columns]
      reg_correct = reg_general[correlation.columns]

      scaler = MinMaxScaler()
      scaled = scaler.fit_transform(reg_correct)

      bkm = BisectingKMeans(n_clusters=10)
      bkm.fit(scaled)

      with_clusters = reg_correct.copy()
      with_clusters['group'] = bkm.labels_
      grouping = with_clusters.groupby(['group'])
      df_mean = {lbl[0]: data.mean() for lbl, data in grouping}
      all_mean = pd.DataFrame(df_mean).transpose()
      qd_df = (all_mean - reg_correct.mean()) ** 2
      qd_df['group'] = all_mean['group']
      max_df = qd_df.max()
      dict_feat = {}
      for col in all_mean.columns:
         dict_feat[col] = list(qd_df[qd_df[col] == max_df[col]]['group'])
      round_df = max_df.round()

      groups_feats = {}
      for lbl, groups in dict_feat.items():
         for group in groups:
            if group in groups_feats:
                  groups_feats[group].append(lbl)
            else:
                  groups_feats[group] = [lbl]
      print(groups_feats)

      def show_graphic_2d(data, labels):
         fig, axs = plt.subplots(1, 1, figsize=(9, 9))
         axs.scatter(data[:, 0], data[:, 1], c=labels, s=2)
         return fig

      def show_graphic_3d(data, labels):
         fig = plt.figure(figsize=(9, 9))
         axs = fig.add_subplot(projection='3d')
         axs.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=1)
         return fig

      def draw_graphic(*args, **kwargs):
         print(*args)
         print(**kwargs)
         fig = show_graphic_2d(embedded_pca, bkm.labels_)
         st.pyplot(fig)
         fig = show_graphic_3d(embedded_pca_3d, bkm.labels_)
         st.pyplot(fig)

      profile = st.selectbox(
         ("Выберите профиль"),
         list(groups_feats.keys()),
         index=None,
         placeholder="Выберите профиль",
         on_change=draw_graphic,
      )

      embedded_pca = PCA(n_components=2).fit_transform(scaled) # точки для 2d
      embedded_pca_3d = PCA(n_components=3).fit_transform(scaled) # точки для 3d
      # bkm.labels_ - цвета для точек

      selected_feats = {}
      for lbl, value in zip(round_df.axes[0], round_df):
         try:
            selected_feats[lbl] = coders[lbl].inverse_transform([int(value)])[0]
         except:
            pass
      print(selected_feats)
