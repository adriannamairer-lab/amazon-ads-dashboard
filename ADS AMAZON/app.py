import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import io
from PIL import Image
import numpy as np
import json
from collections import defaultdict

# 🖥️ Page Configuration
st.set_page_config(page_title="Amazon Advertising Dashboard", layout="wide")

# --- Inicjalizacja stanu sesji ---
if 'rules' not in st.session_state:
    try:
        with open("rules.json", 'r', encoding='utf-8') as f:
            st.session_state.rules = json.load(f)
            # Konwersja starych reguł do nowego formatu
            for rule in st.session_state.rules:
                if 'type' not in rule:
                    rule['type'] = 'Bid'
                if 'value_type' not in rule:
                    rule['value_type'] = 'Wpisz wartość'
                if 'color' not in rule:
                    rule['color'] = 'Czerwony'
    except FileNotFoundError:
        st.session_state.rules = [{"type": "Bid", "name": "", "metric": "ACOS", "condition": "Większe niż", "value": 0.0, "change": 0.0, "value_type": "Wpisz wartość", "color": "Czerwony"} for _ in range(5)]

if 'manual_bid_updates' not in st.session_state:
    st.session_state.manual_bid_updates = None
if 'new_bid_data' not in st.session_state:
    st.session_state.new_bid_data = {}
# --- KONIEC ---


# --- START CSS SECTION ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
.stApp { background-color: #F7F5EF !important; }
h1, h3 { color: #333; }
[data-testid="stTabs"] { margin-top: -3rem !important; }
.main .block-container { padding-top: 2rem; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stAgGrid"]), [data-testid="stVerticalBlock"]:has(> [data-testid="stDataFrame"]) {
    background-color: white; padding: 1.5rem; border-radius: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.05);
}
.main-filter-container [data-testid="stSelectbox"] > div > div {
    background-color: #F45D48 !important; border-radius: 10px !important; border: none !important; color: white !important;
}
.main-filter-container [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #F45D48 !important; border-radius: 10px !important;
}
.main-filter-container [data-testid="stSelectbox"] div, .main-filter-container [data-testid="stSelectbox"] input { color: white !important; }
.main-filter-container [data-testid="stSelectbox"] svg { fill: white !important; }
div[data-baseweb="popover"] li { background-color: #F45D48 !important; color: white !important; }
div[data-baseweb="popover"] li:hover { background-color: #ff7b68 !important; }
.stRadio > label { font-size: 1.1rem; font-weight: 600; color: #333; padding-bottom: 10px; }
.stTextInput input { background-color: white !important; border: 1px solid #E0E0E0 !important; border-radius: 8px; height: 42px; }
div[data-testid="stHorizontalBlock"] { align-items: center; }
p.filter-label { font-weight: 600; color: #333; margin: 0; text-align: right; padding-right: 10px; }
</style>
""", unsafe_allow_html=True)
# --- END CSS SECTION ---


# --- CONFIGURATION AND CONSTANTS ---
KRAJE_MAPA = {
    "Niemcy":   {"tab_id": "10fe_YpeNoM8LqelCf9RQ3uUnQgl6CmnCRm1y6xklBLg", "SP_TYDZIEN": "0", "SP_MIESIAC": "1199906508", "SB_TYDZIEN": "1257325307", "SB_MIESIAC": "11375362", "SD_TYDZIEN": "8043683", "SD_MIESIAC": "304910120", "AK_TYDZIEN": "797863318", "AK_MIESIAC": "2039975432"},
    "Włochy":    {"tab_id": "1F1oz4DTU0XCHy3KHWorIzkpgJKLpDtL3VFO0o4h49_U", "SP_TYDZIEN": "0", "SP_MIESIAC": "1199906508", "SB_TYDZIEN": "1257325307", "SB_MIESIAC": "11375362", "SD_TYDZIEN": "8043683", "SD_MIESIAC": "304910120", "AK_TYDZIEN": "797863318", "AK_MIESIAC": "2039975432"},
    "Francja":  {"tab_id": "1wU7xCI89Nu4sxCNtrYMo9XtVzP30Xjo1MRbQTL-GBOk", "SP_TYDZIEN": "0", "SP_MIESIAC": "1199906508", "SB_TYDZIEN": "1257325307", "SB_MIESIAC": "11375362", "SD_TYDZIEN": "8043683", "SD_MIESIAC": "304910120", "AK_TYDZIEN": "797863318", "AK_MIESIAC": "2039975432"},
    "Hiszpania": {"tab_id": "1AbKJ8fm1fg8aFu9gxATvI2xc1UQmgBd1FaNLDYQ84L8", "SP_TYDZIEN": "0", "SP_MIESIAC": "1199906508", "SB_TYDZIEN": "1257325307", "SB_MIESIAC": "11375362", "SD_TYDZIEN": "8043683", "SD_MIESIAC": "304910120", "AK_TYDZIEN": "797863318", "AK_MIESIAC": "2039975432"},
    "Holandia":  {"tab_id": "16FneIVf1KdN_QKF8LKJNcq5KC3bt8RcJwTo7q9fFR4Y", "SP_TYDZIEN": "0", "SP_MIESIAC": "1199906508", "SB_TYDZIEN": "1257325307", "SB_MIESIAC": "11375362", "SD_TYDZIEN": "8043683", "SD_MIESIAC": "304910120", "AK_TYDZIEN": "797863318", "AK_MIESIAC": "2039975432"},
    "Belgia":    {"tab_id": "1W0NmnYVfWNylNu6QGJBEjI_HADKZiqZNI0cHhVpepe4", "SP_TYDZIEN": "0", "SP_MIESIAC": "1199906508", "SB_TYDZIEN": "1257325307", "SB_MIESIAC": "11375362", "SD_TYDZIEN": "8043683", "SD_MIESIAC": "304910120", "AK_TYDZIEN": "797863318", "AK_MIESIAC": "2039975432"},
    "Polska":    {"tab_id": "15_JPk20Zu3Jk_-AaMQhmJGUTnLm0HfJGF7_Rf1aWu3Q", "SP_TYDZIEN": "0", "SP_MIESIAC": "1199906508", "SB_TYDZIEN": "1257325307", "SB_MIESIAC": "11375362", "SD_TYDZIEN": "8043683", "SD_MIESIAC": "304910120", "AK_TYDZIEN": "797863318", "AK_MIESIAC": "2039975432"}
}

NUMERIC_COLS = ["Spend","Sales","Orders","Daily budget","Impressions","Clicks","CTR","Bid","Bid_new","Price","Quantity","ACOS","CPC","ROAS","Units"]


# --- HELPER FUNCTIONS ---

def get_url(tab_id, gid):
    return f"https://docs.google.com/spreadsheets/d/{tab_id}/export?format=csv&gid={gid}"

@st.cache_data
def load_price_data():
    url = "https://docs.google.com/spreadsheets/d/1Ds_SbZ3Ilg9KbipNyj-FP0V5Bb2mZFmUWoLvRhqxDCA/export?format=csv&gid=1384320249"
    try:
        df = pd.read_csv(url, header=0)
        
        if df.shape[1] < 3:
            st.warning("Arkusz cen nie ma wystarczającej liczby kolumn do pobrania cen i nazw produktów.")
            price_name_map_df = pd.DataFrame(columns=['SKU', 'Price', 'Nazwa produktu'])
        else:
            price_name_map_df = df.iloc[:, [0, 1, 2]].copy()
            price_name_map_df.columns = ['SKU', 'Price', 'Nazwa produktu']
        
        price_name_map_df.dropna(subset=['SKU'], inplace=True)
        price_name_map_df['SKU'] = price_name_map_df['SKU'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

        if df.shape[1] < 29:
             st.warning("Arkusz cen nie ma wystarczającej liczby kolumn do pobrania ilości (oczekiwano co najmniej 29).")
             qty_map_df = pd.DataFrame(columns=['SKU', 'Quantity'])
        else:
            qty_map_df = df.iloc[:, [27, 28]].copy()
            qty_map_df.columns = ['SKU', 'Quantity']
            qty_map_df.dropna(subset=['SKU'], inplace=True)
            qty_map_df['SKU'] = qty_map_df['SKU'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        final_df = pd.merge(price_name_map_df, qty_map_df, on='SKU', how='outer')

        final_df['Price'] = pd.to_numeric(final_df['Price'].astype(str).str.replace(',', '.'), errors='coerce')
        final_df['Quantity'] = pd.to_numeric(final_df['Quantity'], errors='coerce')
        if 'Nazwa produktu' in final_df.columns:
            final_df['Nazwa produktu'] = final_df['Nazwa produktu'].astype(str).fillna('')
        
        final_df = final_df.drop_duplicates(subset=['SKU'], keep='first')
        return final_df

    except Exception as e:
        st.error(f"Błąd podczas ładowania danych o cenach: {e}")
        return pd.DataFrame(columns=['SKU', 'Price', 'Quantity', 'Nazwa produktu'])


def save_rules_to_file(rules, filepath="rules.json"):
    with open(filepath, 'w', encoding='utf-8') as f: json.dump(rules, f, ensure_ascii=False, indent=4)

def clean_numeric_columns(df):
    df_clean = df.copy()
    for c in NUMERIC_COLS:
        if c == "Price (DE)" or c not in df_clean.columns:
            continue
        cleaned_series = df_clean[c].astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        if any(metric in c for metric in ["ACOS", "CTR"]):
            cleaned_series = cleaned_series.str.replace("%", "", regex=False)
        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
        if any(metric in c for metric in ["ACOS", "CTR"]):
            if not numeric_series.empty and pd.notna(numeric_series).any() and numeric_series.max() > 1:
                numeric_series /= 100
        df_clean[c] = numeric_series
    return df_clean

def apply_rules_to_bids_vectorized(df, rules):
    df = df.copy()
    if "Bid" not in df.columns: return df, 0
    bid_loc = df.columns.get_loc("Bid")
    if "Bid_new" not in df.columns:
        df.insert(bid_loc + 1, "Bid_new", df["Bid"])
    else:
        df["Bid_new"] = df["Bid"]
    
    bid_rules = [r for r in rules if r.get('type') == 'Bid']
    active_rules = [r for r in bid_rules if r.get("name") and r.get("value") is not None and r.get("change") is not None]
    if not active_rules: return df, 0
    
    grouped_rules = defaultdict(list)
    for rule in active_rules: grouped_rules[rule['name']].append(rule)
    bids_changed_mask = pd.Series(False, index=df.index)
    
    condition_map = {"Większe niż": ">", "Mniejsze niż": "<", "Równe": "="}

    for name, rule_group in grouped_rules.items():
        final_group_mask = pd.Series(True, index=df.index)
        for rule in rule_group:
            metric = rule.get("metric")
            condition_text = rule.get("condition")
            condition = condition_map.get(condition_text)
            rule_value = rule.get("value")

            if not all([metric, condition, rule_value is not None]) or metric not in df.columns:
                final_group_mask = pd.Series(False, index=df.index); break
            
            row_values = df[metric]
            rule_value_conv = float(rule_value) / 100.0 if metric in ["ACOS", "CTR"] else float(rule_value)
            
            condition_mask = False
            if condition == '>': condition_mask = row_values.gt(rule_value_conv)
            elif condition == '<': condition_mask = row_values.lt(rule_value_conv)
            elif condition == '=': condition_mask = row_values.eq(rule_value_conv)
            
            if isinstance(condition_mask, pd.Series): final_group_mask &= condition_mask
            else: final_group_mask = pd.Series(False, index=df.index); break

        if not final_group_mask.any(): continue
        change_value = rule_group[0].get("change")
        mask_to_apply = final_group_mask & ~bids_changed_mask & df['Bid'].notna() & (df['Bid'] > 0)
        if not mask_to_apply.any() or change_value is None: continue
        try:
            multiplier = 1 + (float(change_value) / 100.0)
            if multiplier < 0: multiplier = 0
            df.loc[mask_to_apply, "Bid_new"] = df.loc[mask_to_apply, "Bid"] * multiplier
            bids_changed_mask.loc[mask_to_apply] = True
        except (ValueError, TypeError): continue
    
    if "Bid_new" in df.columns: df['Bid_new'] = df['Bid_new'].round(2)
    num_changed = int(bids_changed_mask.sum())
    return df, num_changed

def infer_targeting_from_name(campaign_name):
    if not isinstance(campaign_name, str): return "Manual"
    return "Auto" if "AUTO" in campaign_name.upper() else "Manual"

def find_first_existing_column(df, potential_names):
    for name in potential_names:
        if name in df.columns:
            return name
    return None

def process_loaded_data(df_raw, typ_kampanii_arg):
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    
    df = df_raw.copy()
    
    key_cols_to_fix = { "SKU": ["SKU"], "ASIN": ["ASIN (Informational only)"] }
    for col_name, potential_names in key_cols_to_fix.items():
        col_to_fix = find_first_existing_column(df, potential_names)
        if col_to_fix:
            df[col_to_fix] = df[col_to_fix].fillna('').astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            
    df = clean_numeric_columns(df)
    
    if 'Spend' in df.columns and 'Sales' in df.columns:
        df['ACOS'] = np.where(df['Sales'] > 0, df['Spend'] / df['Sales'], 0)
    
    STANDARDIZED_CAMPAIGN_COL = "_Campaign_Standardized"
    POTENTIAL_CAMPAIGN_COLS = ["Campaign name (Informational only)", "Campaign name", "Campaign"]
    actual_campaign_col = find_first_existing_column(df, POTENTIAL_CAMPAIGN_COLS)
    if actual_campaign_col:
        df.rename(columns={actual_campaign_col: STANDARDIZED_CAMPAIGN_COL}, inplace=True)
    
    if 'Clicks' in df.columns and 'Impressions' in df.columns and 'CTR' not in df.columns:
        df['CTR'] = np.where(df['Impressions'] > 0, df['Clicks'] / df['Impressions'], 0)
    if STANDARDIZED_CAMPAIGN_COL in df.columns:
        df["Targeting type"] = df[STANDARDIZED_CAMPAIGN_COL].apply(infer_targeting_from_name)
    
    df['Campaign Type'] = typ_kampanii_arg
        
    return df

@st.cache_data
def load_all_product_data():
    POTENTIAL_CAMPAIGN_NAMES = ["Campaign name (Informational only)", "Campaign name", "Campaign"]
    all_dfs, COLUMN_MAPPING, reports_to_load = [], {'campaign': POTENTIAL_CAMPAIGN_NAMES, 'sku': ["SKU"]}, []
    for country, settings in KRAJE_MAPA.items():
        gids_map = {"Sponsored Products": (settings.get("SP_TYDZIEN"), settings.get("SP_MIESIAC")), "Sponsored Brands": (settings.get("SB_TYDZIEN"), settings.get("SB_MIESIAC")), "Sponsored Display": (settings.get("SD_TYDZIEN"), settings.get("SD_MIESIAC"))}
        for camp_type, gids in gids_map.items():
            for gid in gids:
                if gid: reports_to_load.append({'country': country, 'gid': gid, 'tab_id': settings["tab_id"], 'campaign_type': camp_type})
    if not reports_to_load: return pd.DataFrame()
    for report in reports_to_load:
        try:
            df_temp = pd.read_csv(get_url(report['tab_id'], report['gid']), low_memory=False)
            found_cols = {std: find_first_existing_column(df_temp, poss) for std, poss in COLUMN_MAPPING.items()}
            if not all(found_cols.values()): continue
            df_small = pd.DataFrame({std.capitalize(): df_temp[act] for std, act in found_cols.items()})
            df_small["Country"], df_small["Campaign Type"] = report['country'], report['campaign_type']
            all_dfs.append(df_small)
        except Exception: pass
    if not all_dfs: return pd.DataFrame()
    master_df = pd.concat(all_dfs, ignore_index=True)
    if "Sku" in master_df.columns: master_df["Sku"] = master_df["Sku"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if 'Campaign' in master_df.columns:
        master_df['Targeting Type'] = master_df['Campaign'].apply(infer_targeting_from_name)
        master_df.dropna(subset=["Sku", 'Campaign'], inplace=True)
    return master_df.fillna("")

@st.cache_data
def load_search_data():
    ad_data = load_all_product_data()
    if ad_data.empty:
        return pd.DataFrame()
    product_details = load_price_data()
    if product_details.empty or 'Nazwa produktu' not in product_details.columns:
        ad_data['Nazwa produktu'] = ''
        return ad_data
    if 'Sku' in ad_data.columns and 'SKU' in product_details.columns:
        merged_df = pd.merge(ad_data, product_details[['SKU', 'Nazwa produktu']], left_on='Sku', right_on='SKU', how='left')
        if 'SKU' in merged_df.columns and 'Sku' in merged_df.columns and 'SKU' != 'Sku':
            merged_df = merged_df.drop(columns=['SKU'])
    else:
        merged_df = ad_data
        merged_df['Nazwa produktu'] = ''
    merged_df['Nazwa produktu'].fillna('', inplace=True)
    return merged_df

# --- HEADER ---
header_cols = st.columns([1, 5])
with header_cols[0]:
    try: st.image(Image.open("logo.png"), width=200)
    except FileNotFoundError: st.error("Nie znaleziono pliku logo.png.")
with header_cols[1]:
    st.markdown("<h1 style='margin-top: 25px; margin-left: -40px; color: #333; font-size: 48px;'>AMAZON ADVERTISING DASHBOARD</h1>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 New Bid", "🔍 Find Product ID", "📜 Rules"])

with tab1:
    # Funkcja pomocnicza do obliczania podsumowania, aby uniknąć powtarzania kodu
    def calculate_summary_row(df, currency_symbol):
        if df.empty:
            return None
            
        summary_data = {}
        cols_to_sum = ['Impressions', 'Clicks', 'Spend', 'Orders', 'Sales', 'Daily budget']
        existing_cols_to_sum = [col for col in cols_to_sum if col in df.columns]
        sums = df[existing_cols_to_sum].sum()

        for col in existing_cols_to_sum:
            summary_data[col] = sums.get(col, 0)

        sum_clicks, sum_impressions, sum_spend, sum_sales = sums.get('Clicks', 0), sums.get('Impressions', 0), sums.get('Spend', 0), sums.get('Sales', 0)
        
        summary_data['CTR'] = (sum_clicks / sum_impressions) if sum_impressions > 0 else 0
        summary_data['CPC'] = (sum_spend / sum_clicks) if sum_clicks > 0 else 0
        summary_data['ACOS'] = (sum_spend / sum_sales) if sum_sales > 0 else 0
        summary_data['ROAS'] = (sum_sales / sum_spend) if sum_spend > 0 else 0

        # Dodajemy logikę dla okresu porównawczego
        if 'Spend_M' in df.columns:
            cols_m_to_sum = [f"{col}_M" for col in cols_to_sum if f"{col}_M" in df.columns]
            sums_m = df[cols_m_to_sum].sum()
            for col_m in cols_m_to_sum: summary_data[col_m] = sums_m.get(col_m, 0)
            sum_clicks_m, sum_impressions_m, sum_spend_m, sum_sales_m = sums_m.get('Clicks_M', 0), sums_m.get('Impressions_M', 0), sums_m.get('Spend_M', 0), sums_m.get('Sales_M', 0)
            summary_data['CTR_M'] = (sum_clicks_m / sum_impressions_m) if sum_impressions_m > 0 else 0
            summary_data['CPC_M'] = (sum_spend_m / sum_clicks_m) if sum_clicks_m > 0 else 0
            summary_data['ACOS_M'] = (sum_spend_m / sum_sales_m) if sum_sales_m > 0 else 0
            summary_data['ROAS_M'] = (sum_sales_m / sum_spend_m) if sum_spend_m > 0 else 0


        first_col_name = df.columns[0]
        summary_data[first_col_name] = "SUMA"
        
        return pd.DataFrame([summary_data]).to_dict('records')

    with st.spinner("Przetwarzam dane..."):
        filter_container = st.container()
        with filter_container:
            main_cols = st.columns([1.5, 2, 0.5, 3])
            with main_cols[0]:
                okres = st.radio("Przedział czasowy:", ["Tydzień", "Miesiąc", "Porównanie"])
            with main_cols[1]:
                st.markdown('<div class="main-filter-container">', unsafe_allow_html=True)
                kraj = st.selectbox("Kraj", list(KRAJE_MAPA.keys()))
                typ_kampanii = st.selectbox("Typ kampanii", ["Sponsored Products", "Sponsored Brands", "Sponsored Display"])
                
                if typ_kampanii == "Sponsored Products":
                    widok_options = ["Campaign", "Product ad", "Product targeting", "Keyword", "Auto keyword/ASIN"]
                elif typ_kampanii == "Sponsored Brands":
                    widok_options = ["Campaign", "Keyword", "Product targeting"]
                elif typ_kampanii == "Sponsored Display":
                    widok_options = ["Campaign", "Product ad", "Product targeting", "Audience targeting", "Contextual targeting"]
                else:
                    widok_options = ["Campaign", "Product ad", "Product targeting", "Keyword", "Audience targeting", "Contextual targeting"]
                
                widok = st.selectbox("Widok", widok_options)
                st.markdown('</div>', unsafe_allow_html=True)
            with main_cols[3]:
                st.markdown("<h4 style='text-align: center; font-weight: 600; color: #333; margin-bottom: 17px;'>Filtrowanie</h4>", unsafe_allow_html=True)
                def render_metric_filter(label, key):
                    cols = st.columns([1, 1.5, 1.4])
                    cols[0].markdown(f"<p class='filter-label'>{label}</p>", unsafe_allow_html=True)
                    op = cols[1].selectbox(f"{key}_op", ["Brak", ">", "<", "="], key=f"{key}_op", label_visibility="collapsed")
                    val = cols[2].text_input(f"{key}_val", key=f"{key}_value", label_visibility="collapsed")
                    return op, val
                spend_filter, spend_value = render_metric_filter("Spend", "spend")
                sales_filter, sales_value = render_metric_filter("Sales", "sales")
                orders_filter, orders_value = render_metric_filter("Orders", "orders")
                acos_filter, acos_value = render_metric_filter("ACOS", "acos")
                roas_filter, roas_value = render_metric_filter("ROAS", "roas")
                ctr_filter, ctr_value = render_metric_filter("CTR", "ctr")
                
                highlight_filter_placeholder = st.empty()

        st.markdown("<br>", unsafe_allow_html=True)
        ustawienia_kraju = KRAJE_MAPA[kraj]
        tab_id = ustawienia_kraju["tab_id"]

        gid_w, gid_m = None, None
        if typ_kampanii == "Sponsored Products":
            gid_w = ustawienia_kraju.get("AK_TYDZIEN") if widok == "Auto keyword/ASIN" else ustawienia_kraju.get("SP_TYDZIEN")
            gid_m = ustawienia_kraju.get("AK_MIESIAC") if widok == "Auto keyword/ASIN" else ustawienia_kraju.get("SP_MIESIAC")
        elif typ_kampanii == "Sponsored Brands":
            gid_w, gid_m = ustawienia_kraju.get("SB_TYDZIEN"), ustawienia_kraju.get("SB_MIESIAC")
        elif typ_kampanii == "Sponsored Display":
            gid_w, gid_m = ustawienia_kraju.get("SD_TYDZIEN"), ustawienia_kraju.get("SD_MIESIAC")
        
        df_w_raw, df_m_raw = None, None
        try:
            if okres in ["Tydzień", "Porównanie"] and gid_w:
                df_w_raw = pd.read_csv(get_url(tab_id, gid_w))
            if okres in ["Miesiąc", "Porównanie"] and gid_m:
                df_m_raw = pd.read_csv(get_url(tab_id, gid_m))
        except Exception as e:
            st.error(f"Błąd ładowania danych źródłowych: {e}")

        df_w = process_loaded_data(df_w_raw, typ_kampanii)
        df_m = process_loaded_data(df_m_raw, typ_kampanii)

        if okres == "Tydzień": df = df_w
        elif okres == "Miesiąc": df = df_m
        else: df = df_w

        if not df.empty:
            STANDARDIZED_CAMPAIGN_COL = "_Campaign_Standardized"
            base = {
                "Campaign": STANDARDIZED_CAMPAIGN_COL, "Match type": "Match type", 
                "Keyword text": "Keyword text", "Targeting expression": "Product targeting expression",
                "SKU": "SKU", "Customer search term": "Customer search term", "Targeting type":"Targeting type",
                "Product":"Product","Portfolio":"Portfolio name (Informational only)","Entity":"Entity","State":"State",
                "ASIN":"ASIN (Informational only)", "Nazwa produktu": "Nazwa produktu",
                "Price": "Price", "Quantity": "Quantity", "Daily budget":"Daily budget",
                "Campaign Type": "Campaign Type", "Impressions":"Impressions", "Clicks":"Clicks", "CTR": "CTR",
                "Spend":"Spend", "CPC":"CPC", "Orders":"Orders", "Sales":"Sales", "ACOS":"ACOS", "ROAS":"ROAS", "Bid":"Bid"
            }

            if widok == "Product ad":
                ordered = ["Campaign", "Targeting type", "Campaign Type", "Entity", "State", "SKU", "ASIN", "Nazwa produktu", "Price", "Quantity", "Impressions", "Clicks", "CTR", "Spend", "CPC", "Orders", "Sales", "ACOS", "ROAS", "Bid"]
            else:
                ordered = ["Campaign", "Match type", "Keyword text", "Targeting expression", "Customer search term", "Targeting type", "Product", "Portfolio", "Entity", "State", "Daily budget", "Impressions", "Clicks", "CTR", "Spend", "CPC", "Orders", "Sales", "ACOS", "ROAS", "Bid"]
                if widok == "Product targeting":
                    if "Match type" in ordered: ordered.remove("Match type")
                    if "Product targeting expression" in df.columns: df["Match type"] = df["Product targeting expression"].astype(str).str.split('=', n=1).str[0]
                if widok == "Auto keyword/ASIN" and 'Match type' in df.columns:
                    df = df[df['Match type'].isna()].copy()
            
            if widok == "Product ad":
                prices_df = load_price_data()
                if not prices_df.empty:
                    sku_col_name = find_first_existing_column(df, ["SKU"])
                    if sku_col_name:
                        df = pd.merge(df, prices_df, on=sku_col_name, how='left')
            
            cols_map = {k: v for k in ordered for v in [base.get(k)] if v and v in df.columns}
            ordered_final = [k for k in ordered if k in cols_map]
            if 'Entity' in df.columns: df = df[df["Entity"] == widok]

            if okres == "Porównanie" and not df_m.empty:
                numeric_metrics = [c for c in NUMERIC_COLS if c not in ['Bid', 'Bid_new', 'Daily budget']]
                key_display_names = [k for k in ordered_final if k not in numeric_metrics]
                key_source_names = list(set([cols_map[k] for k in key_display_names if k in cols_map]))
                
                metric_display_names = [k for k in ordered_final if k in numeric_metrics]
                metric_source_names = list(set([cols_map[k] for k in metric_display_names if k in cols_map]))

                monthly_cols_to_keep = [c for c in key_source_names + metric_source_names if c in df_m.columns]
                df_m_prepared = df_m[monthly_cols_to_keep].copy()
                
                rename_dict = {col: f"{col}_M" for col in metric_source_names}
                df_m_prepared.rename(columns=rename_dict, inplace=True)

                if key_source_names and not df_m_prepared.empty:
                    df = pd.merge(df, df_m_prepared, on=key_source_names, how='left')

                ordered_interleaved = []
                for col_name in ordered_final:
                    ordered_interleaved.append(col_name)
                    if col_name in metric_display_names:
                        ordered_interleaved.append(f"{col_name}_M")
                ordered_final = ordered_interleaved
            
            df_display_builder = {}
            for col_display_name in ordered_final:
                if col_display_name in cols_map:
                    col_source_name = cols_map[col_display_name]
                    if col_source_name in df.columns:
                        df_display_builder[col_display_name] = df[col_source_name]
                elif col_display_name in df.columns:
                    df_display_builder[col_display_name] = df[col_display_name]
            df_display = pd.DataFrame(df_display_builder)

            df_display_rules, _ = apply_rules_to_bids_vectorized(df_display, st.session_state.rules)
            
            filter_map = {
                "Spend": (spend_filter, spend_value), "Sales": (sales_filter, sales_value), 
                "Orders": (orders_filter, orders_value), "ACOS": (acos_filter, acos_value), 
                "ROAS": (roas_filter, roas_value), "CTR": (ctr_filter, ctr_value)
            }
            for col, (op, val_str) in filter_map.items():
                if op != "Brak" and val_str and col in df_display_rules.columns:
                    try:
                        val = float(val_str)
                        if col in ['ACOS', 'CTR']: val /= 100.0
                        if op == ">": df_display_rules = df_display_rules[df_display_rules[col] > val]
                        elif op == "<": df_display_rules = df_display_rules[df_display_rules[col] < val]
                        elif op == "=": df_display_rules = df_display_rules[df_display_rules[col] == val]
                    except (ValueError, TypeError, KeyError): pass
            
            summary_for_dropdown_df = pd.DataFrame(calculate_summary_row(df_display_rules, 'EUR'))

            highlight_rules = [r for r in st.session_state.rules if r.get('type') == 'Highlight' and r.get('name')]
            highlight_filter_options = {"Brak": None}
            for rule in highlight_rules:
                value_display = ""
                if rule['value_type'] == 'Średnia z konta':
                    if summary_for_dropdown_df is not None and rule['metric'] in summary_for_dropdown_df.columns:
                        avg_val = summary_for_dropdown_df.iloc[0][rule['metric']]
                        if rule['metric'] in ['ACOS', 'CTR']: value_display = f"Średnia ({avg_val:.2%})"
                        else: value_display = f"Średnia ({avg_val:.2f})"
                    else: value_display = "Średnia z konta"
                else:
                    value_display = str(rule.get('value', ''))
                    if rule['metric'] in ['ACOS', 'CTR']: value_display += "%"
                option_label = f"{rule['name']} ({rule['metric']} {rule['condition']} {value_display})"
                highlight_filter_options[option_label] = rule['name']
            
            with highlight_filter_placeholder.container():
                selected_label = st.selectbox("Filtruj wg podświetlenia:", list(highlight_filter_options.keys()))
            
            selected_rule_name = highlight_filter_options[selected_label]
            if selected_rule_name:
                selected_rule = next((r for r in highlight_rules if r['name'] == selected_rule_name), None)
                if selected_rule:
                    metric, condition_text = selected_rule['metric'], selected_rule['condition']
                    threshold = 0.0
                    if selected_rule['value_type'] == 'Średnia z konta':
                        if summary_for_dropdown_df is not None and metric in summary_for_dropdown_df.columns: threshold = summary_for_dropdown_df.iloc[0][metric]
                    else:
                        threshold = float(selected_rule.get('value', 0.0))
                        if metric in ['ACOS', 'CTR']: threshold /= 100.0
                    
                    if condition_text == "Większe niż": df_display_rules = df_display_rules[df_display_rules[metric] > threshold]
                    elif condition_text == "Mniejsze niż": df_display_rules = df_display_rules[df_display_rules[metric] < threshold]
                    elif condition_text == "Równe": df_display_rules = df_display_rules[df_display_rules[metric] == threshold]

            if not df_display_rules.empty:
                cols_to_drop = [col for col in df_display_rules.columns if df_display_rules[col].isnull().all()]
                df_display_rules.drop(columns=cols_to_drop, inplace=True)
            if "_Campaign_Standardized" in df_display_rules.columns:
                df_display_rules.rename(columns={"_Campaign_Standardized": "Campaign"}, inplace=True)

            gb = GridOptionsBuilder.from_dataframe(df_display_rules)
            
            js_conditions, condition_map_js, color_map_hex = [], {"Większe niż": ">", "Mniejsze niż": "<", "Równe": "==="}, {"Zielony": "#d4edda", "Pomarańczowy": "#fff3cd", "Czerwony": "#f8d7da"}
            if highlight_rules and summary_for_dropdown_df is not None:
                for rule in highlight_rules:
                    metric, condition_symbol, color_name = rule['metric'], condition_map_js.get(rule['condition']), rule.get('color', 'Czerwony')
                    color_hex = color_map_hex.get(color_name, '#f8d7da')
                    value = 0
                    if rule.get('value_type') == 'Średnia z konta':
                        if metric in summary_for_dropdown_df.columns: value = summary_for_dropdown_df.iloc[0][metric]
                    else:
                        value = rule.get('value', 0.0)
                        if metric in ["ACOS", "CTR"]: value /= 100.0
                    if condition_symbol:
                        js_conditions.append(f"if (params.colDef.field === '{metric}' && params.value {condition_symbol} {value}) {{ return {{'backgroundColor': '{color_hex}'}}; }}")
            
            js_pinned_row_style = JsCode("""function(params) { if (params.node.rowPinned) { return {'fontWeight': 'bold', 'backgroundColor': '#f0f2f6'}; } }""")
            cell_style_jscode = JsCode(f"function(params) {{ {' '.join(js_conditions)} if (params.node.rowPinned) {{ return {{'fontWeight': 'bold', 'backgroundColor': '#f0f2f6'}}; }} return null; }}")
            gb.configure_default_column(resizable=True, autoHeaderHeight=True, cellStyle=cell_style_jscode)
            
            for col in df_display_rules.columns:
                gb.configure_column(col, filter='agSetColumnFilter')
            
            if "Bid_new" in df_display_rules.columns: gb.configure_column("Bid_new", editable=True)
            currency_symbol = 'PLN' if kraj == 'Polska' else 'EUR'
            for c in df_display_rules.columns:
                c_base = c.replace('_M', '')
                formatter_js = None
                if c_base in NUMERIC_COLS or c == "Campaign":
                    if "ACOS" in c_base or "CTR" in c_base: formatter_js=JsCode("""function(params){if(params.value==null||isNaN(params.value))return'';return params.value.toLocaleString('pl-PL',{style:'percent', minimumFractionDigits:2, maximumFractionDigits:2})}""")
                    elif any(x in c_base for x in["Price","Spend","Sales","Bid","CPC"]): formatter_js=JsCode(f"""function(params){{if(params.value==null||isNaN(params.value))return'';return params.value.toLocaleString('pl-PL',{{style:'currency',currency:'{currency_symbol}',maximumFractionDigits:2}})}}""")
                    elif "ROAS" in c_base: formatter_js=JsCode("""function(params){if(params.value==null||isNaN(params.value))return'';return params.value.toLocaleString('pl-PL',{maximumFractionDigits:2})}""")
                    elif c != "Campaign": formatter_js=JsCode("""function(params){if(params.value==null||isNaN(params.value))return'';return Math.round(params.value).toLocaleString('pl-PL')}""")
                    if c != "Campaign": gb.configure_column(c, type=["numericColumn","rightAligned"], valueFormatter=formatter_js)

            grid_key = f"{kraj}_{typ_kampanii}_{widok}_{okres}"
            if 'current_grid_key' not in st.session_state or st.session_state.current_grid_key != grid_key:
                st.session_state.current_grid_key = grid_key
                st.session_state.pinned_row = calculate_summary_row(df_display_rules, currency_symbol)
                st.session_state.grid_state = {} 

            grid_options = gb.build()
            grid_options['pinnedBottomRowData'] = st.session_state.pinned_row
            grid_options['getRowStyle'] = js_pinned_row_style

            grid_response = AgGrid(
                df_display_rules, 
                gridOptions=grid_options, 
                update_mode=GridUpdateMode.FILTERING_CHANGED | GridUpdateMode.SORTING_CHANGED,
                grid_state=st.session_state.get('grid_state', {}),
                width="100%", 
                height=600, 
                allow_unsafe_jscode=True, 
                theme='ag-theme-alpine',
                key=grid_key,
                # ZMIANA: Przywracamy moduły Enterprise, co jest kluczowe dla działania filtra
                enable_enterprise_modules=True
            )
            
            st.session_state.grid_state = grid_response['grid_state']

            df_filtered = pd.DataFrame(grid_response['data'])
            newly_calculated_row = calculate_summary_row(df_filtered, currency_symbol)

            if newly_calculated_row != st.session_state.pinned_row:
                st.session_state.pinned_row = newly_calculated_row
                st.rerun()

            st.markdown("---")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_filtered.to_excel(writer, index=False, sheet_name='DashboardExport')
                workbook, worksheet = writer.book, writer.sheets['DashboardExport']
                header = df_filtered.columns.values.tolist()
                currency_format = workbook.add_format({'num_format': f'#,##0.00 "{currency_symbol}"'}); percent_format = workbook.add_format({'num_format': '0.00%'}); integer_format = workbook.add_format({'num_format': '#,##0'}); roas_format = workbook.add_format({'num_format': '#,##0.00'})
                for idx, col_name in enumerate(header):
                    c_base = col_name.replace('_M', '')
                    if "ACOS" in c_base or "CTR" in c_base: worksheet.set_column(idx, idx, 12, percent_format)
                    elif any(x in c_base for x in ["Price", "Spend", "Sales", "Bid", "CPC"]): worksheet.set_column(idx, idx, 15, currency_format)
                    elif "ROAS" in c_base: worksheet.set_column(idx, idx, 12, roas_format)
                    elif c_base in ["Orders", "Impressions", "Clicks", "Quantity", "Units"]: worksheet.set_column(idx, idx, 12, integer_format)
                worksheet.autofit()
            
            st.download_button(label="📥 Pobierz Excel", data=buffer, file_name=f"dashboard_{kraj}_{typ_kampanii}_{widok}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", help="Pobierz dane widoczne w tabeli jako plik Excel")

            st.markdown("---")
            if st.button("✅ Zapisz ręczne zmiany do zastosowania w 'New Bid'", use_container_width=True):
                if grid_response['data'] is not None:
                    updated_df = pd.DataFrame(grid_response['data'])
                    st.session_state.manual_bid_updates = { "data": updated_df, "widok": widok, "kraj": kraj, "typ_kampanii": typ_kampanii, "cols_map": cols_map }
                    st.success("Ręczne zmiany stawek zostały zapisane! Zostaną one automatycznie nałożone w zakładce 'New Bid'.")
                else:
                    st.warning("Brak danych w tabeli do zapisania.")
        else:
            st.warning("Brak danych dla wybranego okresu lub filtrów.")

with tab2:
    st.header("Automatycznie zaktualizowany plik z nowymi stawkami")
    st.info("Dane w tej tabeli są aktualizowane automatycznie. Najpierw stosowane są reguły, a następnie nakładane są zapisane zmiany z Dashboardu.")
    kraj_nb = st.selectbox("Kraj", list(KRAJE_MAPA.keys()), key="newbid_kraj_4")
    typ_kampanii_nb = st.selectbox("Typ kampanii", ["Sponsored Products", "Sponsored Brands", "Sponsored Display"], key="newbid_typ_4")
    state_key = f"{kraj_nb}_{typ_kampanii_nb}"
    gid_key = {"Sponsored Products": "SP_TYDZIEN", "Sponsored Brands": "SB_TYDZIEN", "Sponsored Display": "SD_TYDZIEN"}.get(typ_kampanii_nb)
    gid = KRAJE_MAPA[kraj_nb].get(gid_key)
    tab_id = KRAJE_MAPA[kraj_nb]["tab_id"]
    if gid:
        try:
            with st.spinner("Przetwarzanie danych..."):
                base_df_raw = pd.read_csv(get_url(tab_id, gid), dtype=str)
                original_columns = base_df_raw.columns.tolist()
                if 'Operation' not in original_columns:
                    original_columns.append('Operation')
                    base_df_raw['Operation'] = ''

                base_df_cleaned = clean_numeric_columns(base_df_raw)
                if 'Spend' in base_df_cleaned.columns and 'Sales' in base_df_cleaned.columns:
                    base_df_cleaned['ACOS'] = np.where(base_df_cleaned['Sales'] > 0, base_df_cleaned['Spend'] / base_df_cleaned['Sales'], 0)
                df_with_rules, num_rules_applied = apply_rules_to_bids_vectorized(base_df_cleaned, st.session_state.rules)
                st.success(f"Krok 1: Zastosowano reguły automatyczne, zmieniając {num_rules_applied} stawek.")
                updates = st.session_state.get('manual_bid_updates')
                final_df_processed = df_with_rules
                key_cols_map = {}
                target_keys = []
                if updates and updates['kraj'] == kraj_nb and updates['typ_kampanii'] == typ_kampanii_nb:
                    updates_df = updates['data']
                    source_cols_map = updates.get("cols_map", {})
                    widok_source = updates['widok']
                    key_clean_names = ['Campaign', 'Ad Group']
                    if widok_source == 'Keyword': key_clean_names.extend(['Keyword text', 'Match type'])
                    elif widok_source in ['Product targeting', 'Audience targeting', 'Contextual targeting']: key_clean_names.append('Targeting expression')
                    elif widok_source == 'Product ad': key_clean_names.append('SKU')
                    elif widok_source == 'Auto keyword/ASIN': key_clean_names.append('Customer search term')
                    key_cols_map = {clean: source_cols_map.get(clean) for clean in key_clean_names if source_cols_map.get(clean)}
                    target_keys = list(key_cols_map.values())
                    if key_cols_map and target_keys and 'Bid_new' in updates_df.columns:
                        if all(k in final_df_processed.columns for k in target_keys):
                            updates_renamed = updates_df.rename(columns=key_cols_map)
                            updates_for_merge = updates_renamed[target_keys + ['Bid_new']].copy()
                            updates_for_merge.rename(columns={'Bid_new': 'Bid_new_manual'}, inplace=True)
                            updates_for_merge['Bid_new_manual'] = pd.to_numeric(updates_for_merge['Bid_new_manual'], errors='coerce')
                            updates_for_merge.dropna(subset=['Bid_new_manual'], inplace=True)
                            for key in target_keys:
                                final_df_processed[key] = final_df_processed[key].fillna('')
                                updates_for_merge[key] = updates_for_merge[key].fillna('')
                            merged_df = pd.merge(final_df_processed, updates_for_merge, on=target_keys, how='left')
                            merged_df['Bid_new'] = merged_df['Bid_new_manual'].fillna(merged_df['Bid_new'])
                            final_df_processed = merged_df.drop(columns=['Bid_new_manual'])
                            st.success(f"Krok 2: Pomyślnie nałożono {len(updates_for_merge)} ręcznych zmian z Dashboardu.")
                
                final_df_processed['Bid'] = final_df_processed['Bid_new']
                final_df_processed['Bid'] = final_df_processed['Bid'].apply(lambda x: '' if pd.isna(x) else str(x))

                final_df_for_download = base_df_raw.copy()
                if target_keys:
                    update_payload = final_df_processed[target_keys + ['Bid']].copy()
                    for key in target_keys:
                        final_df_for_download[key] = final_df_for_download[key].fillna('')
                        update_payload[key] = update_payload[key].fillna('')
                    df_to_update = final_df_for_download.set_index(target_keys)
                    payload_indexed = update_payload.set_index(target_keys)
                    df_to_update.update(payload_indexed)
                    final_df_for_download = df_to_update.reset_index()
                else: 
                    final_df_for_download['Bid'] = final_df_processed['Bid'].values
                
                if 'Bid' in final_df_for_download.columns and 'Operation' in final_df_for_download.columns:
                    final_df_for_download['Operation'] = np.where(final_df_for_download['Bid'].astype(str).str.strip().ne(''), 'Update', '')
                
                st.session_state.new_bid_data[state_key] = final_df_for_download[original_columns]
        except Exception as e:
            st.error(f"Wystąpił błąd: {e}")
    if state_key in st.session_state.new_bid_data:
        display_df = st.session_state.new_bid_data[state_key]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            display_df.to_excel(writer, index=False, sheet_name='Updated_Bids')
        st.download_button(label="📥 Pobierz plik z nowymi stawkami (.xlsx)", data=buffer.getvalue(), file_name=f"NewBids_{kraj_nb}_{typ_kampanii_nb}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab3:
    st.header("Wyszukiwarka Produktów")
    search_df = load_search_data()

    if search_df.empty:
        st.error("Nie udało się załadować danych o produktach.")
    else:
        search_term = st.text_input("Wpisz SKU, nazwę produktu (można wiele po przecinku) i naciśnij Enter:")

        if search_term:
            search_list = [term.strip() for term in search_term.split(',') if term.strip()]
            
            if search_list:
                combined_mask = pd.Series([False] * len(search_df), index=search_df.index)
                
                for term in search_list:
                    term_mask = (
                        search_df['Sku'].str.contains(term, case=False, na=False) |
                        search_df['Nazwa produktu'].str.contains(term, case=False, na=False)
                    )
                    combined_mask = combined_mask | term_mask
                
                results_df = search_df[combined_mask]

                if not results_df.empty:
                    display_cols = ['Sku', 'Nazwa produktu', 'Campaign', 'Campaign Type', 'Targeting Type', 'Country']
                    display_cols_exist = [col for col in display_cols if col in results_df.columns]
                    
                    display_df = results_df[display_cols_exist].drop_duplicates().sort_values(by=['Sku', 'Country', 'Campaign'])
                    st.markdown(f"--- \n**Znaleziono {len(display_df)} unikalnych wyników dla:** `{', '.join(search_list)}`")
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        display_df.to_excel(writer, index=False, sheet_name='SearchResults')
                        writer.sheets['SearchResults'].autofit()

                    st.download_button(
                        label="📥 Pobierz wyniki do Excela",
                        data=buffer,
                        file_name=f"product_search_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    gb = GridOptionsBuilder.from_dataframe(display_df)
                    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
                    AgGrid(display_df, gridOptions=gb.build(), height=600, width='100%', allow_unsafe_jscode=True, theme='streamlit')
                else:
                    st.warning(f"Nie znaleziono żadnych produktów pasujących do: '{', '.join(search_list)}'.")
        else:
            st.info("Wpisz SKU lub nazwę, aby rozpocząć wyszukiwanie.")


with tab4:
    st.header("📜 Konfiguracja Reguł Optymalizacji Stawek")
    st.markdown("---")
    st.info("""Zdefiniuj reguły do automatycznej zmiany stawek (Bid) lub wizualnego wyróżniania komórek w tabeli (Highlight).""")
    
    header_cols = st.columns([1.5, 2, 2, 1.5, 2.5, 2, 0.5])
    header_cols[0].markdown("**Typ reguły**"); header_cols[1].markdown("**Nazwa Reguły**"); header_cols[2].markdown("**Wskaźnik**")
    header_cols[3].markdown("**Warunek**"); header_cols[4].markdown("**Wartość**"); header_cols[5].markdown("**Akcja / Kolor**")
    st.divider()

    rules_to_render = list(st.session_state.rules)
    for i, rule in enumerate(rules_to_render):
        with st.container():
            cols = st.columns([1.5, 2, 2, 1.5, 2.5, 2, 0.5])
            
            rule_type = cols[0].selectbox("Typ", ["Bid", "Highlight"], index=["Bid", "Highlight"].index(rule.get("type", "Bid")), key=f"type_{i}", label_visibility="collapsed")
            st.session_state.rules[i]["type"] = rule_type

            st.session_state.rules[i]["name"] = cols[1].text_input("Nazwa", value=rule.get("name", ""), key=f"name_{i}", label_visibility="collapsed")
            
            metrics_list = ["ACOS", "ROAS", "CTR", "Spend", "Sales", "Orders", "Impressions", "Clicks", "CPC"]
            metric_val = rule.get("metric", "ACOS")
            metric_idx = metrics_list.index(metric_val) if metric_val in metrics_list else 0
            st.session_state.rules[i]["metric"] = cols[2].selectbox("Wskaźnik", options=metrics_list, index=metric_idx, key=f"metric_{i}", label_visibility="collapsed")
            
            condition_list_text = ['Większe niż', 'Mniejsze niż', 'Równe']
            condition_map_rev = {'>':'Większe niż', '<':'Mniejsze niż', '=':'Równe'}
            condition_val = condition_map_rev.get(rule.get("condition"), rule.get("condition", "Większe niż"))
            condition_idx = condition_list_text.index(condition_val) if condition_val in condition_list_text else 0
            st.session_state.rules[i]["condition"] = cols[3].selectbox("Założenie", options=condition_list_text, index=condition_idx, key=f"condition_{i}", label_visibility="collapsed")

            if rule_type == "Bid":
                with cols[4]:
                    if st.session_state.rules[i]["metric"] in ["ACOS", "CTR"]:
                        val_cols = st.columns([3, 1])
                        value = val_cols[0].number_input("Wartość", value=float(rule.get("value", 0.0)), key=f"value_{i}", format="%.2f", label_visibility="collapsed")
                        val_cols[1].markdown("<p style='margin-top: 8px;'>%</p>", unsafe_allow_html=True)
                        st.session_state.rules[i]["value"] = value
                    else:
                        st.session_state.rules[i]["value"] = st.number_input("Wartość", value=float(rule.get("value", 0.0)), key=f"value_{i}", format="%.2f", label_visibility="collapsed")
                with cols[5]:
                    change_cols = st.columns([3, 1])
                    change_value = change_cols[0].number_input("Zmiana %", value=float(rule.get("change", 0.0)), key=f"change_{i}", format="%.2f", label_visibility="collapsed")
                    change_cols[1].markdown("<p style='margin-top: 8px;'>%</p>", unsafe_allow_html=True)
                    st.session_state.rules[i]["change"] = change_value
            
            elif rule_type == "Highlight":
                with cols[4]:
                    value_type = st.selectbox("Typ wartości", ["Wpisz wartość", "Średnia z konta"], index=["Wpisz wartość", "Średnia z konta"].index(rule.get("value_type", "Wpisz wartość")), key=f"valuetype_{i}", label_visibility="collapsed")
                    st.session_state.rules[i]["value_type"] = value_type
                    
                    if value_type == "Wpisz wartość":
                        if st.session_state.rules[i]["metric"] in ["ACOS", "CTR"]:
                            val_cols = st.columns([3, 1])
                            value = val_cols[0].number_input("Wartość %", value=float(rule.get("value", 0.0)), key=f"value_{i}", format="%.2f", label_visibility="collapsed")
                            val_cols[1].markdown("<p style='margin-top: 8px;'>%</p>", unsafe_allow_html=True)
                            st.session_state.rules[i]["value"] = value
                        else:
                            st.session_state.rules[i]["value"] = st.number_input("Wartość", value=float(rule.get("value", 0.0)), key=f"value_{i}", format="%.2f", label_visibility="collapsed")
                    else:
                        st.markdown("<div style='height: 38px; display: flex; align-items: center; justify-content: center; color: #888; background-color: #fafafa; border-radius: 8px;'>Średnia z konta</div>", unsafe_allow_html=True)
                        st.session_state.rules[i]["value"] = 0.0
                with cols[5]:
                    color_options = ["Czerwony", "Pomarańczowy", "Zielony"]
                    color_val = rule.get("color", "Czerwony")
                    color_idx = color_options.index(color_val) if color_val in color_options else 0
                    st.session_state.rules[i]["color"] = st.selectbox("Kolor", options=color_options, index=color_idx, key=f"color_{i}", label_visibility="collapsed")
                    st.session_state.rules[i]["change"] = 0.0
            
            if cols[6].button("🗑️", key=f"delete_{i}", help="Usuń tę regułę"):
                st.session_state.rules.pop(i); st.rerun()

    st.divider()
    action_cols = st.columns([1, 1, 2])
    num_to_add = action_cols[0].number_input("Liczba wierszy do dodania:", min_value=1, value=1, step=1)
    if action_cols[1].button(f"➕ Dodaj wiersz(e)", use_container_width=True):
        for _ in range(num_to_add): st.session_state.rules.append({"type":"Bid", "name": "", "metric": "ACOS", "condition": "Większe niż", "value": 0.0, "change": 0.0, "value_type": "Wpisz wartość", "color": "Czerwony"})
        st.rerun()
    if action_cols[2].button("🧹 Wyczyść nieaktywne reguły", use_container_width=True, help="Usuwa wszystkie reguły, które nie mają nazwy"):
        st.session_state.rules = [r for r in st.session_state.rules if r.get("name")]; st.rerun()
    
    with st.expander("Podgląd zapisanych reguł (dane surowe)"):
        st.dataframe(pd.DataFrame([r for r in st.session_state.rules if r.get("name")]))
    
    save_rules_to_file(st.session_state.rules)