# ==============================================================
# Fusion 7.4 Interactive ER Studio – Complete Streamlit Application
# ==============================================================

import streamlit as st
import pandas as pd
import sqlite3
import hashlib, datetime, re
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from fpdf import FPDF
import plotly.express as px
from streamlit_lottie import st_lottie
import requests, seaborn as sns, matplotlib.pyplot as plt

try:
    from graphviz import Digraph
    graphviz_available = True
except ImportError:
    graphviz_available = False

st.set_page_config(page_title="Fusion 7.4 Interactive ER Studio", layout="wide")

# ---------- Core Utilities ----------
def make_hash(p): return hashlib.sha256(p.encode()).hexdigest()

@st.cache_resource
def init_db():
    conn = sqlite3.connect("fusion_core.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY, password TEXT, created_at TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS uploads(
        username TEXT, table_name TEXT, upload_date TEXT, rows INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS insights(
        username TEXT, insight_text TEXT, created_at TEXT)""")
    conn.commit()
    return conn, cur
conn, cur = init_db()

def register_user(u,p):
    cur.execute("INSERT OR IGNORE INTO users VALUES (?,?,?)",
                (u, make_hash(p), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

def verify_user(u,p):
    cur.execute("SELECT * FROM users WHERE username=? AND password=?",(u,make_hash(p)))
    return cur.fetchone()

def record_upload(u,t,r):
    cur.execute("INSERT INTO uploads VALUES (?,?,?,?)",
                (u,t,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),r))
    conn.commit()

def save_insight(u,text):
    cur.execute("INSERT INTO insights VALUES (?,?,?)",
                (u,text,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

def get_tables(cur):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [t[0] for t in cur.fetchall()]

def clean_query(q):  # Fix invisible Unicode spaces
    return re.sub(r'[\u202f\u200b\u00a0]', ' ', q).strip()

def load_lottieurl(url):
    try:
        r=requests.get(url);
        return r.json() if r.status_code==200 else None
    except: return None

# ---------- LOGIN SCREENS ----------
lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_iwmd6pyr.json")
st.markdown("<style>[data-testid=stAppViewContainer]{background:linear-gradient(135deg,#000428,#004e92);} </style>", unsafe_allow_html=True)

if "username" not in st.session_state:
    menu=["Login","Sign Up"]
    option=st.sidebar.radio("Authentication",menu)
    if option=="Sign Up":
        st.title("🧊 Create SQL Account")
        u,p=st.text_input("Username"),st.text_input("Password",type="password")
        if st.button("Sign Up"): register_user(u,p); st.success("Account created!")
    else:
        st.title("🔐 Login to Fusion Studio")
        if lottie: st_lottie(lottie,height=180)
        u,p=st.text_input("Username"),st.text_input("Password",type="password")
        if st.button("Login"):
            if verify_user(u,p):
                st.session_state["username"]=u
                st.session_state["uploaded_table"]=None
                st.session_state["history"]=[]
                st.session_state["auto_query"]=None
                st.rerun()
            else: st.error("Invalid credentials")

# ---------- MAIN APP ----------
if "username" in st.session_state:
    user=st.session_state["username"]
    db=sqlite3.connect("fusion_core.db",check_same_thread=False)
    c=db.cursor()
    st.sidebar.header(f"👋 Welcome, {user}")
    if st.sidebar.button("Logout"): st.session_state.clear(); st.rerun()
    page=st.sidebar.radio("Sections",["🏠 Upload Data","💾 Studio (SQL + ER)","📊 Analytics","📈 Insights","📊 Dashboard"])

    # ---- Upload ----
    if page.startswith("🏠"):
        st.header("Upload CSV → SQL Table")
        f=st.file_uploader("Upload CSV",type=["csv"])
        if st.button("Load Sample Data"):
            df=pd.DataFrame({"id":[1,2,3,4],"category":["Books","Electronics","Grocery","Stationery"],"sales":[80,120,60,50]})
            tbl=f"{user}_data"; df.to_sql(tbl,db,if_exists="replace",index=False)
            record_upload(user,tbl,len(df)); st.session_state["uploaded_table"]=tbl; st.success(f"Sample data → {tbl}"); st.dataframe(df)
        elif f:
            df=pd.read_csv(f).drop_duplicates().fillna(0); tbl=f"{user}_data"
            df.to_sql(tbl,db,if_exists="replace",index=False)
            record_upload(user,tbl,len(df)); st.session_state["uploaded_table"]=tbl
            st.success(f"✅ Uploaded → {tbl}"); st.dataframe(df.head())

    # ---- SQL Studio ----
    elif page.startswith("💾"):
        st.header("💾 Fusion Studio – ER Diagram + Live SQL")
        left,right=st.columns([2,3])
        with left:
            st.subheader("🧩 ER Diagram")
            if not graphviz_available: st.error("Graphviz not installed.")
            else:
                t=get_tables(c); dot=Digraph(); dot.attr(rankdir="LR",bgcolor="transparent")
                for x in t: dot.node(x,x,shape="box",color="lightblue",style="filled")
                st.graphviz_chart(dot)
                sel=st.selectbox("🖱 Select Table",t)
                if sel: st.session_state["auto_query"]=f"SELECT * FROM {sel};"; st.info(f"Auto‑Query: {sel}")
        with right:
            st.subheader("🧮 SQL Executor")
            q=st.text_area("SQL Editor",st.session_state.get("auto_query") or "SELECT * FROM uploads;")
            if st.button("Run Query"):
                try:
                    res=pd.read_sql_query(clean_query(q),db)
                    st.success(f"{len(res)} rows returned."); st.dataframe(res)
                except Exception as e: st.error(e)

    # ---- Analytics ----
    elif page.startswith("📊 Analytics"):
        st.header("📊 Exploratory Data Analytics")
        tbl=st.session_state.get("uploaded_table",f"{user}_data")
        try:
            df=pd.read_sql_query(f"SELECT * FROM {tbl}",db)
            if df.empty: st.warning("Upload data first.")
            else:
                nums=df.select_dtypes('number').columns
                if len(nums):
                    col=st.selectbox("Numeric Column",nums)
                    st.metric("Total",f"{df[col].sum():,.2f}")
                    st.metric("Average",f"{df[col].mean():,.2f}")
                    st.plotly_chart(px.histogram(df,x=col))
                    st.plotly_chart(px.box(df,y=col,points="all"))
                    st.subheader("Correlation Heatmap")
                    fig,ax=plt.subplots(); sns.heatmap(df[nums].corr(),annot=True,cmap="coolwarm",ax=ax); st.pyplot(fig)
                    st.subheader("Descriptive Statistics"); st.dataframe(df.describe())
        except Exception as e: st.warning(f"Upload data first. ({e})")

    # ---- Insights ----
    elif page.startswith("📈"):
        st.header("📈 AI Generated Insights and PDF")
        tbl=st.session_state.get("uploaded_table",f"{user}_data")
        try:
            df=pd.read_sql_query(f"SELECT * FROM {tbl}",db)
            num=df.select_dtypes('number').columns
            lines=[]
            if len(num):
                total,avg=df[num[0]].sum(),df[num[0]].mean()
                lines+=[f"Total={total:,.2f}",f"Average={avg:,.2f}"]
            cats=[c for c in df.columns if "category" in c.lower()]
            if cats and len(num):
                top=df.groupby(cats[0])[num[0]].sum().nlargest(3)
                lines.append("Top Categories:"); [lines.append(f"{k}:{v:,.2f}") for k,v in top.items()]
            txt="\n".join(lines); st.text_area("Insight Summary",txt,height=230); save_insight(user,txt)
            if st.button("📄 Export PDF"):
                pdf=FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
                [pdf.multi_cell(0,10,x) for x in lines]; fn=f"{user}_Insights_{datetime.date.today()}.pdf"
                pdf.output(fn); open(fn,"rb")
                with open(fn,"rb") as f: st.download_button("Download PDF",f,file_name=fn)
        except Exception as e: st.warning(f"Upload data first. ({e})")

    # ---- Dashboard ----
    elif page.startswith("📊 Dashboard"):
        st.header("📊 Executive Dashboard + ML Predictions")
        tbl=st.session_state.get("uploaded_table",f"{user}_data")
        try:
            df=pd.read_sql_query(f"SELECT * FROM {tbl}",db)
            nums,objs=df.select_dtypes('number').columns,df.select_dtypes('object').columns
            if df.empty: st.warning("No data available.")
            else:
                st.sidebar.subheader("🔍 Filters")
                if len(objs):
                    cat=st.sidebar.selectbox("Category Column",objs)
                    val=st.sidebar.multiselect("Select Values",df[cat].unique(),df[cat].unique())
                    df=df[df[cat].isin(val)]
                if len(nums):
                    c1,c2,c3=st.columns(3)
                    c1.metric("Rows",str(len(df)))
                    c2.metric("Average",f"{df[nums[0]].mean():.2f}")
                    c3.metric("Max",f"{df[nums[0]].max():.2f}")
                    sel=st.selectbox("Metric",nums)
                    st.plotly_chart(px.line(df,y=sel,title=f"Trend of {sel}"))
                    if len(objs): st.plotly_chart(px.bar(df,x=cat,y=sel,color=cat))
                st.subheader("🤖 Predictive Model‑Linear Regression")
                if len(nums)>=2:
                    target=st.selectbox("Target",nums)
                    features=st.multiselect("Features",[n for n in nums if n!=target],
                                             default=[n for n in nums if n!=target])
                    if st.button("Train Model"):
                        X,y=df[features],df[target]
                        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
                        m=LinearRegression().fit(Xtr,ytr); preds=m.predict(Xte)
                        mse=mean_squared_error(yte,preds); rmse=np.sqrt(mse); r2=r2_score(yte,preds)
                        st.success(f"Trained → RMSE={rmse:.3f}, R²={r2:.3f}")
                        comp=pd.DataFrame({"Actual":yte,"Predicted":preds})
                        st.plotly_chart(px.scatter(comp,x="Actual",y="Predicted",trendline="ols"))
        except Exception as e: st.warning("Upload data first."); st.error(e)
