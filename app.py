import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import io 

# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Bilinéarisation Pushover",
    page_icon="🏗️",
    layout="wide"
)

# ==============================================================================
# MODULE 0 — SÉCURISATION ET CHARGEMENT DES DONNÉES
# ==============================================================================
def charger_donnees(fichier_excel):
    df = pd.read_excel(fichier_excel)
    
    col_d = [c for c in df.columns if str(c).lower().startswith('d')]
    col_f = [c for c in df.columns if str(c).lower().startswith('f')]

    if not col_d or not col_f:
        raise ValueError("Colonnes introuvables. Le fichier doit avoir une colonne commençant par 'd' et une par 'f'.")

    df_clean = pd.DataFrame()
    df_clean['d'] = pd.to_numeric(df[col_d[0]], errors='coerce')
    df_clean['f'] = pd.to_numeric(df[col_f[0]], errors='coerce')
    df_clean = df_clean.dropna()

    if df_clean.empty:
        raise ValueError("Le fichier ne contient aucune donnée numérique valide après nettoyage.")

    d = df_clean['d'].values
    f = df_clean['f'].values

    idx = np.argsort(d)
    d, f = d[idx], f[idx]

    d, uniq = np.unique(d, return_index=True)
    f = f[uniq]

    return d, f

# ==============================================================================
# MODULE 1 — LISSAGE SAVITZKY-GOLAY
# ==============================================================================
def lissage_donnees(d, f, window_length=None, polyorder=3):
    n = len(f)
    if window_length is None:
        wl = max(5, int(n * 0.10))
        window_length = wl + (1 - wl % 2)
    window_length = max(window_length, polyorder + 2)
    if window_length % 2 == 0:
        window_length += 1

    f_lisse = savgol_filter(f, window_length=window_length, polyorder=polyorder)
    f_lisse = np.maximum(f_lisse, 0.0)
    return f_lisse

# ==============================================================================
# MODULE 2 — BILINÉARISATION EUROCODE 8
# ==============================================================================
def bilinearisation_eurocode8(d, f_lisse):
    idx_max = np.argmax(f_lisse)
    Vy      = f_lisse[idx_max]
    d_m     = d[idx_max]

    masque = d <= d_m
    E_m    = trapezoid(f_lisse[masque], d[masque])
    
    dy = 2.0 * (d_m - E_m / Vy)

    if dy <= 0.0:
        raise ValueError("dy <= 0 : courbe trop rigide pour EC8.")
    if dy >= d_m:
        raise ValueError("dy >= d_m : comportement quasi-élastique.")

    Ke = Vy / dy
    aire_bilin = Vy * d_m - 0.5 * Vy * dy
    diff_pct   = abs(E_m - aire_bilin) / E_m * 100.0

    d_bilin = np.array([0.0, dy, d.max()])
    f_bilin = np.array([0.0, Vy, Vy])
    
    mu = d_m / dy 

    return {
        "Ke": Ke, "Vy": Vy, "dy": dy, "d_m": d_m, "E_m": E_m,
        "aire_bilin": aire_bilin, "diff_pct": diff_pct,
        "d_bilin": d_bilin, "f_bilin": f_bilin, "mu": mu
    }

# ==============================================================================
# MODULE 3 — BILINÉARISATION ASCE 41-23
# ==============================================================================
def bilinearisation_asce41(d, f_lisse, delta_t=None, tol=1e-4, max_iter=300):
    f_de_d = interp1d(d, f_lisse, kind='linear', bounds_error=False, fill_value=(f_lisse[0], f_lisse[-1]))

    idx_pic = np.argmax(f_lisse)
    Vmax    = f_lisse[idx_pic]
    d_pic   = d[idx_pic]
    
    f_mont  = f_lisse[:idx_pic + 1]
    d_mont  = d[:idx_pic + 1]
    f_uniq, idx_uniq = np.unique(f_mont, return_index=True)
    d_inv = interp1d(f_uniq, d_mont[idx_uniq], kind='linear', bounds_error=False, fill_value=(d_mont[0], d_mont[-1]))

    if delta_t is None or delta_t == 0.0:
        delta_t = d.max()
    Delta_d = min(delta_t, d_pic)
    Vd_cible = float(f_de_d(Delta_d))

    Vy = Vmax
    converge = False
    
    # Variables pour stocker l'énergie
    E_reel_fin = 0
    aire_BL_fin = 0

    for it in range(max_iter):
        F60 = 0.60 * Vy
        Delta60 = float(d_inv(F60))
        Delta60 = max(Delta60, d[1])
        Ke = F60 / Delta60
        Delta_y = Vy / Ke
        Vd = Vd_cible

        masque_dd = d <= Delta_d
        E_reel = trapezoid(f_lisse[masque_dd], d[masque_dd])
        aire_BL = (0.5 * Vy * Delta_y + 0.5 * (Vy + Vd) * (Delta_d - Delta_y))

        Vy_new = Vy + (E_reel - aire_BL) / (0.5 * Delta_d)
        Vy_new = min(max(Vy_new, 0.01 * Vmax), Vmax)

        if abs(Vy_new - Vy) < tol:
            Vy = Vy_new
            converge = True
            E_reel_fin = E_reel
            aire_BL_fin = aire_BL
            break
        Vy = Vy_new

    F60_f = 0.60 * Vy
    Delta60_f = max(float(d_inv(F60_f)), d[1])
    Ke = F60_f / Delta60_f
    Delta_y = Vy / Ke

    denom1 = Ke * max(Delta_d - Delta_y, 1e-12)
    alpha1 = (Vd - Vy) / denom1

    seuil_degrad = 0.60 * Vy
    d_desc = d[idx_pic:]
    f_desc = f_lisse[idx_pic:]

    if len(d_desc) > 2 and f_desc[-1] < seuil_degrad:
        f_desc_uniq, idx_desc = np.unique(f_desc[::-1], return_index=True)
        d_desc_r = d_desc[::-1][idx_desc]
        d_inv_desc = interp1d(f_desc_uniq, d_desc_r, kind='linear', bounds_error=False, fill_value=(d_desc_r[0], d_desc_r[-1]))
        Delta_degrad = float(d_inv_desc(seuil_degrad))
        alpha2 = (seuil_degrad - Vd) / (Ke * max(Delta_degrad - Delta_d, 1e-12))
    else:
        Delta_degrad = Delta_d + abs(Vd - seuil_degrad) / max(0.05 * Ke, 1e-10)
        alpha2 = (seuil_degrad - Vd) / (Ke * max(Delta_degrad - Delta_d, 1e-12))

    d_bilin = np.array([0.0, Delta_y, Delta_d, Delta_degrad])
    f_bilin = np.array([0.0, Vy, Vd, seuil_degrad])
    
    mu = Delta_d / Delta_y
    diff_pct = abs(E_reel_fin - aire_BL_fin) / max(E_reel_fin, 1e-10) * 100.0

    return {
        "Ke": Ke, "Vy": Vy, "dy": Delta_y, "Delta_d": Delta_d,
        "alpha1": alpha1, "alpha2": alpha2, "F60": F60_f, "Delta60": Delta60_f,
        "d_bilin": d_bilin, "f_bilin": f_bilin, "mu": mu, "converge": converge,
        "E_reel": E_reel_fin, "aire_BL": aire_BL_fin, "diff_pct": diff_pct
    }

# ==============================================================================
# MODULE 4 — TRACÉ QUALITÉ PUBLICATION
# ==============================================================================
def tracer_graphique_publication(d, f_lisse, d_bilin, f_bilin, dy, Vy, dm, res_dict, methode="EC8"):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    ax.plot(d, f_lisse, color='#1565C0', lw=2.5, label="Courbe lissée (Savgol)")
    
    label_bilin = f"Bilinéaire {methode}\n(Ke = {res_dict['Ke']:.1f}, μ = {res_dict['mu']:.2f})"
    ax.plot(d_bilin, f_bilin, color='#C62828', lw=2.5, ls='--', label=label_bilin)
    
    ax.plot(dy, Vy, marker='o', markersize=10, markeredgecolor='black', markerfacecolor='#FFC107', 
            lw=0, label=f"Point de fluage (dy={dy:.2f})")
    
    if methode == "EC8":
        ax.plot(dm, Vy, marker='s', markersize=10, markeredgecolor='black', markerfacecolor='#4CAF50', 
                lw=0, label=f"Point ultime (dm={dm:.2f})")
    else:
        ax.plot(res_dict['Delta_d'], res_dict['f_bilin'][2], marker='s', markersize=10, markeredgecolor='black', markerfacecolor='#4CAF50', 
                lw=0, label=f"Point cible (Δd={res_dict['Delta_d']:.2f})")

    ax.set_xlabel("Déplacement [mm]", fontweight='bold')
    ax.set_ylabel("Force de cisaillement [kN]", fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=True)
    
    fig.tight_layout()
    return fig

# ==============================================================================
# MODULE 5 — CRÉATION DU FICHIER EXCEL DETAILLÉ
# ==============================================================================
def generer_fichier_excel(res_ec8, res_asce, d, f_lisse):
    """Génère un fichier Excel en mémoire avec des coordonnées claires et tous les détails."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        
        # 1. Onglet des Résultats et Paramètres
        df_kpi = pd.DataFrame({
            "Paramètres et Indicateurs": [
                "Raideur Initiale (Ke)",
                "Force de Plastification / Fluage (Vy)",
                "Déplacement Élastique (dy)",
                "Ductilité (μ)",
                "Déplacement Max ou Cible (dm / Δd)",
                "Pente Post-Yield Positive (α1)",
                "Pente de Dégradation Négative (α2)",
                "Énergie dissipée réelle (Aire sous la courbe)",
                "Énergie courbe bilinéaire (Aire bilinéaire)",
                "Marge d'erreur sur l'énergie (%)"
            ],
            "Eurocode 8": [
                round(res_ec8['Ke'], 3),
                round(res_ec8['Vy'], 3),
                round(res_ec8['dy'], 3),
                round(res_ec8['mu'], 3),
                round(res_ec8['d_m'], 3),
                "0.0 (Élasto-Plastique Parfait)",
                "N/A",
                round(res_ec8['E_m'], 3),
                round(res_ec8['aire_bilin'], 3),
                round(res_ec8['diff_pct'], 4)
            ],
            "ASCE 41-23": [
                round(res_asce['Ke'], 3),
                round(res_asce['Vy'], 3),
                round(res_asce['dy'], 3),
                round(res_asce['mu'], 3),
                round(res_asce['Delta_d'], 3),
                round(res_asce['alpha1'], 5),
                round(res_asce['alpha2'], 5),
                round(res_asce['E_reel'], 3),
                round(res_asce['aire_BL'], 3),
                round(res_asce['diff_pct'], 4)
            ]
        })
        df_kpi.to_excel(writer, sheet_name="Résultats Détaillés", index=False)
        
        # 2. Onglet Coordonnées EC8
        df_ec8 = pd.DataFrame({
            "Description du Point": ["Point 0 (Origine)", "Point 1 (Fluage effectif)", "Point 2 (Déplacement ultime)"],
            "Déplacement d (mm)": res_ec8["d_bilin"],
            "Force V (kN)": res_ec8["f_bilin"]
        })
        df_ec8.to_excel(writer, sheet_name="Coordonnées_EC8", index=False)
        
        # 3. Onglet Coordonnées ASCE 41
        df_asce = pd.DataFrame({
            "Description du Point": ["Point 0 (Origine)", "Point 1 (Fluage effectif)", "Point 2 (Déplacement cible)", "Point 3 (Dégradation 60% Vy)"],
            "Déplacement d (mm)": res_asce["d_bilin"],
            "Force V (kN)": res_asce["f_bilin"]
        })
        df_asce.to_excel(writer, sheet_name="Coordonnées_ASCE", index=False)
        
        # 4. Onglet Courbe de Base Lissée
        df_lisse = pd.DataFrame({
            "Déplacement Brut/Lissé (mm)": d,
            "Force Lissée (kN)": f_lisse
        })
        df_lisse.to_excel(writer, sheet_name="Courbe_Continue_Lissée", index=False)

    return output.getvalue()

# ==============================================================================
# INTERFACE UTILISATEUR STREAMLIT
# ==============================================================================

st.title("🏗️ Application de Bilinéarisation de la Courbe Pushover")

with st.expander("📖 Guide méthodologique & Rappels théoriques"):
    st.markdown("""
    **Principe Général** : La bilinéarisation consiste à remplacer la courbe de capacité réelle par une courbe idéalisée.
    
    **1. Eurocode 8 (Annexe B.3)** :
    - Repose sur le principe de l'**équivalence des aires** jusqu'au déplacement correspondant à la force maximale ($d_m$).
    
    **2. ASCE 41-23 (§7.4.3.2.5)** :
    - Détermine la raideur initiale $K_e$ par une **sécante à 60%** de la force de plastification effective ($V_y$).
    - Procédure itérative convergeant vers l'égalité des aires.
    """)

st.markdown("---")

st.sidebar.header("📁 Importation des données")
uploaded_file = st.sidebar.file_uploader("Chargez votre fichier Excel (d-f.xlsx)", type=["xlsx"])
delta_t_input = st.sidebar.number_input("Déplacement cible (ASCE) - Laisser 0.0 pour auto", value=0.0, step=1.0)

if uploaded_file is not None:
    try:
        d_brut, f_brut = charger_donnees(uploaded_file)
        f_lisse = lissage_donnees(d_brut, f_brut)
        
        res_ec8 = bilinearisation_eurocode8(d_brut, f_lisse)
        
        dt_val = delta_t_input if delta_t_input > 0 else None
        res_asce = bilinearisation_asce41(d_brut, f_lisse, delta_t=dt_val)

        if not res_asce["converge"]:
            st.warning("⚠️ Attention : La méthode ASCE 41 n'a pas convergé complètement, les résultats peuvent être approximatifs.")

        tab1, tab2 = st.tabs(["📘 Eurocode 8 (Élasto-Plastique)", "📗 ASCE 41-23 (Tri-linéaire)"])

        with tab1:
            st.subheader("📊 Indicateurs de Performance - EC8")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Raideur Initiale (Ke)", f"{res_ec8['Ke']:.2f}")
            col2.metric("Force Max (Vy)", f"{res_ec8['Vy']:.2f}")
            col3.metric("Dép. Élastique (dy)", f"{res_ec8['dy']:.2f}")
            col4.metric("Ductilité (μ)", f"{res_ec8['mu']:.2f}")
            
            fig_ec8 = tracer_graphique_publication(d_brut, f_lisse, res_ec8["d_bilin"], res_ec8["f_bilin"], res_ec8["dy"], res_ec8["Vy"], res_ec8["d_m"], res_ec8, "EC8")
            st.pyplot(fig_ec8)

        with tab2:
            st.subheader("📊 Indicateurs de Performance - ASCE 41-23")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Raideur Initiale (Ke)", f"{res_asce['Ke']:.2f}")
            col2.metric("Force de Fluage (Vy)", f"{res_asce['Vy']:.2f}")
            col3.metric("Pente Post-Yield (α1)", f"{res_asce['alpha1']:.4f}")
            col4.metric("Ductilité (μ)", f"{res_asce['mu']:.2f}")
            
            fig_asce = tracer_graphique_publication(d_brut, f_lisse, res_asce["d_bilin"], res_asce["f_bilin"], res_asce["dy"], res_asce["Vy"], res_asce["Delta_d"], res_asce, "ASCE 41")
            st.pyplot(fig_asce)

        st.success("✅ Calculs effectués avec succès ! Vous pouvez télécharger les données ci-dessous.")

        st.markdown("---")
        st.subheader("📥 Exporter les résultats mathématiques et coordonnées")
        
        # Génération du fichier Excel 
        excel_data = generer_fichier_excel(res_ec8, res_asce, d_brut, f_lisse)
        
        st.download_button(
            label="📊 Télécharger les résultats en Excel (.xlsx)",
            data=excel_data,
            file_name="Resultats_Pushover_Bilinearises.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Télécharge un fichier Excel contenant tous les indicateurs et les coordonnées exactes des points pour vos logiciels."
        )

    except Exception as e:
        st.error(f"❌ Une erreur est survenue lors du traitement. \n\n**Erreur :** {str(e)}")

else:
    st.info("👋 Bienvenue ! Veuillez charger un fichier Excel dans la barre latérale pour commencer.")
