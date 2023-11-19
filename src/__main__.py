import json

import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import streamlit as st

from graphics import load_file, plot_increasing, plot_box, plot_charTS
from ml.model import ModelProphet


st.set_page_config(page_title="Дашборд Сбер")
st.title('Дашборд Сбер')
st.session_state["files"] = {
   "df_general_profile": {"file_id": "", "df": None},
   "df_general_dashboard": {"df": None},
   "df_acquiring_dashboard": {"file_id": "", "df": None},
}

tab1, tab2, tab3 = st.tabs(["Дашборд", "Предсказание", "Профили"])

with tab1:
   st.title("Построение дашбордов")
   uploaded_file_acquiring_dashboard = st.file_uploader(
      "Выберите файл эквайринга",
      key="dashboard_acquiring",
   )

   df_acquiring, df_general = None, None
   st.cache
   if uploaded_file_acquiring_dashboard and uploaded_file_acquiring_dashboard.name.endswith(".xlsb"):
      if (
         st.session_state["files"]['df_general_dashboard']["file_id"] != uploaded_file_acquiring_dashboard.file_id or
         st.session_state["files"]['df_general_dashboard']["df"] is None
      ):
         my_bar = st.progress(0, text="Обработка файла эквайринга")
         df_acquiring = load_file(file_path=uploaded_file_acquiring_dashboard.getvalue(), header=1, need_data_fix=True)
         my_bar.progress(50, text="Обработка файла с основной информацией")
         df_general = load_file(file_path='./src/dataset/general.xlsb', header=0)
         my_bar.progress(100, text="Обработка файлов завершена")
         my_bar.empty()
         st.session_state["files"]['df_general_dashboard']["df"] = df_general
         st.session_state["files"]['df_acquiring_dashboard']["df"] = df_acquiring
         st.session_state["files"]['df_acquiring_dashboard']["file_id"] = uploaded_file_acquiring_dashboard.file_id
      else:
         df_general = st.session_state["files"]['df_general_dashboard']["df"]
         df_acquiring = st.session_state["files"]['df_acquiring_dashboard']["df"]

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
            df=df_acquiring_merged,
            feature=col,
            series=series_cols,
            name='продукта',
         )
         st.pyplot(fig)


      fig = plot_box(data)
      st.pyplot(fig)

      st.subheader("Сезонная декомпозиция")
      fig = plot_charTS(data["y"])
      st.pyplot(fig)

with tab2:
   st.title("Предсказание значений")
   date = st.date_input(
      "Введите дату для предсказания",
      value="today",
      format="YYYY-MM-DD",
      label_visibility="visible",
   )

   model = None

   with st.form(key="form_predict"):
      tasks_predict = [
         "Предсказание роста клиентов",
         "Предсказание оттока клиентов",
         "Предсказание выживаемости клиентов",
      ]
      task = st.selectbox(
         ("Выберите задачу предсказания"),
         tasks_predict,
         index=None,
         placeholder="Выберите задачу предсказания",
      )

      products = ["Эквайринг", "РКО"]
      product = st.selectbox(
         ("Выберите продукт предсказания"),
         products,
         index=None,
         placeholder="Выберите продукт предсказания",
      )
      submit_button = st.form_submit_button(label="Предсказать", type="primary")

      if submit_button:
         if task is not None and product is not None:
            if task == tasks_predict[0]:
               if product == products[0]:
                  model = ModelProphet('model_growth_clients_eq.pkl', 'data_growth_clients_eq.csv', floor=304, cap=47000)
               elif product == products[1]:
                  model = ModelProphet('model_growth_clients_rko.pkl', 'data_growth_clients_rko.csv', floor=304, cap=47000)
            elif task == tasks_predict[1]:
               if product == products[0]:
                  model = ModelProphet('model_cr_eq.pkl', 'data_cr_eq.csv', floor=0, cap=100)
               elif product == products[1]:
                  model = ModelProphet('model_cr_rko.pkl', 'data_cr_rko.csv', floor=0, cap=100)
            elif task == tasks_predict[2]:
               if product == products[0]:
                  model = ModelProphet('model_pr_eq.pkl', 'data_pr_eq.csv', floor=0, cap=100)
               elif product == products[1]:
                  model = ModelProphet('model_pr_rko.pkl', 'data_pr_rko.csv', floor=0, cap=100)
            
            df_final = model.predict(date.year, date.month, date.day)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f'Тестирование модели для {product}')
            ax.set_xlabel("Период")
            true = df_final.iloc[:model.old_date.shape[0]]
            ax.plot(true.ds, true.y, "g", label="true", linewidth=2.0)
            pred = df_final.iloc[model.old_date.shape[0]-1:]
            ax.plot(pred.ds, pred.y, "r", label="prediction", linewidth=2.0)

            ax.set_ylabel(task)
            plt.xticks(rotation=30)
            st.pyplot(fig)

with tab3:
   st.title("Профили решений")

   uploaded_file_general_profile = st.file_uploader(
      "Выберите файл с основной информацией",
      key="profile_general",
   )
   
   df_general = None
  
   if uploaded_file_general_profile and uploaded_file_general_profile.name.endswith(".xlsb"):
      if st.session_state["files"]['df_general_profile']["file_id"] != uploaded_file_general_profile.file_id:
         my_bar = st.progress(0, text="Обработка главного файла")
         df_general = load_file(file_path=uploaded_file_general_profile.getvalue(), header=0)
         my_bar.progress(100, text="Обработка файлов")
         my_bar.empty()
         st.session_state["files"]['df_general_profile']["df"] = df_general
         st.session_state["files"]['df_general_profile']["file_id"] = uploaded_file_general_profile.file_id
      else:
         df_general = st.session_state["files"]['df_general_profile']["df"]

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
      
      round_df = max_df.round()

      dict_feat = {}
      for col in all_mean.columns:
         dict_feat[col] = list(qd_df[qd_df[col] == max_df[col]]['group'].astype('int'))

      groups_feats = {}
      for lbl, groups in dict_feat.items():
         for group in groups:
            if lbl == 'group':
                  continue
            if group in groups_feats:
                  groups_feats[group].append(lbl)
            else:
                  groups_feats[group] = [lbl]

      def show_graphic_2d(data, labels):
         fig, ax = plt.subplots(1, 1, figsize=(9, 9))
         ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
         )
         scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, s=2, label=labels)
         legend = ax.legend(
            *scatter.legend_elements(), loc="best", title="Профиль"
         )
         ax.add_artist(legend)
         return fig

      def show_graphic_3d(data, labels):
         fig = plt.figure(figsize=(9, 9))
         ax = fig.add_subplot(projection='3d')
         scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=1)
         ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
         )
         legend = ax.legend(
            *scatter.legend_elements(), loc="best", title="Профиль"
         )
         ax.add_artist(legend)
         return fig

      with st.form(key="form_profile"):
         profile = st.selectbox(
            ("Выберите профиль"),
            list(groups_feats.keys()),
            placeholder="Выберите профиль",
         )
         submit_button = st.form_submit_button(label="Отрисовать", type="primary")

         if submit_button and profile is not None:
            embedded_pca = PCA(n_components=2).fit_transform(scaled) # точки для 2d
            embedded_pca_3d = PCA(n_components=3).fit_transform(scaled) # точки для 3d
            # bkm.labels_ - цвета для точек

            fig = show_graphic_2d(embedded_pca, bkm.labels_)
            st.pyplot(fig)
            fig = show_graphic_3d(embedded_pca_3d, bkm.labels_)
            st.pyplot(fig)

            selected_feats = {}
            for lbl, value in zip(round_df.axes[0], round_df):
               try:
                  selected_feats[lbl] = coders[lbl].inverse_transform([int(value)])[0]
               except:
                  pass

            with open ('labels.json', 'r', encoding='utf-8') as file:
               labels_dict = json.loads(file.read())

            texts = {}
            for group in range(10):
               if group in groups_feats:
                  chars = set()
                  for feat in groups_feats[group][:4]:
                        value = selected_feats.get(feat, '')
                        if value in labels_dict and feat in labels_dict[value]:
                           chars.add(labels_dict[value][feat])
                        else:
                           chars.add(f'{feat}: {value}')
                  texts[group] = ', '.join(chars)
               else:
                  texts[group] = 'Среднестатистический пользователь'
            
            st.subheader("Характеристика пользователя")

            st.write(texts[profile].capitalize())
