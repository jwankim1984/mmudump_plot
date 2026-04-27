"""
MMU Dump Plot - Streamlit Web Application
DMRS SNR Calculator
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ==================== 3GPP TS 38.211 DMRS Functions ====================

def gold_sequence(c_init, length):
    """Generate Gold sequence (3GPP TS 38.211 Section 5.2.1)"""
    N = length + 1600
    x1 = np.zeros(N + 31, dtype=np.uint32)
    x2 = np.zeros(N + 31, dtype=np.uint32)
    
    x1[0] = 1
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    
    for n in range(31, N + 31):
        x1[n] = (x1[n - 28] + x1[n - 31]) % 2
        x2[n] = (x2[n - 28] + x2[n - 29] + x2[n - 30] + x2[n - 31]) % 2
    
    c = (x1[1600:1600+length] + x2[1600:1600+length]) % 2
    return c

def generate_nr_dmrs_type1_3gpp(n_rb, start_rb, slot, symbol_idx, n_id, n_scid=0, cdm_group=0):
    """Generate NR DMRS sequence for Type 1"""
    m = n_rb * 6
    l = symbol_idx
    c_init = (2**17 * (14 * slot + l + 1) * (2 * n_id + 1) + 2 * n_id + n_scid) % (2**31)
    
    c = gold_sequence(c_init, 2 * m).astype(np.int32)
    
    dmrs_seq = np.zeros(m, dtype=complex)
    for i in range(m):
        i_val = (1 - 2 * c[2*i]) / np.sqrt(2)
        q_val = (1 - 2 * c[2*i + 1]) / np.sqrt(2)
        dmrs_seq[i] = i_val + 1j * q_val
    
    k_prime = cdm_group
    dmrs_indices = []
    for rb in range(n_rb):
        rb_start = (start_rb + rb) * 12
        for m_val in range(6):
            k = rb_start + k_prime + 2 * m_val
            dmrs_indices.append(k)
    
    return dmrs_seq, np.array(dmrs_indices)

def generate_nr_dmrs_type2_3gpp(n_rb, start_rb, slot, symbol_idx, n_id, n_scid=0, cdm_group=0):
    """Generate NR DMRS sequence for Type 2"""
    m = n_rb * 4
    l = symbol_idx
    c_init = (2**17 * (14 * slot + l + 1) * (2 * n_id + 1) + 2 * n_id + n_scid) % (2**31)
    
    c = gold_sequence(c_init, 2 * m).astype(np.int32)
    
    dmrs_seq = np.zeros(m, dtype=complex)
    for i in range(m):
        i_val = (1 - 2 * c[2*i]) / np.sqrt(2)
        q_val = (1 - 2 * c[2*i + 1]) / np.sqrt(2)
        dmrs_seq[i] = i_val + 1j * q_val
    
    dmrs_indices = []
    for rb in range(n_rb):
        rb_start = (start_rb + rb) * 12
        if cdm_group == 0:
            dmrs_indices.extend([rb_start + 0, rb_start + 1])
        elif cdm_group == 1:
            dmrs_indices.extend([rb_start + 6, rb_start + 7])
        elif cdm_group == 2:
            dmrs_indices.extend([rb_start + 2, rb_start + 3, rb_start + 8, rb_start + 9])
    
    return dmrs_seq, np.array(dmrs_indices)

def calculate_dmrs_snr_3gpp(rx_dmrs_symbol, ideal_dmrs, dmrs_indices):
    """Calculate SNR using ideal DMRS sequence"""
    rx_dmrs = rx_dmrs_symbol[dmrs_indices]
    H = rx_dmrs / ideal_dmrs
    H_smooth = np.convolve(H, np.ones(5)/5, mode='same')
    noise_est = H - H_smooth
    noise_power = np.mean(np.abs(noise_est) ** 2)
    signal_power = np.mean(np.abs(H_smooth) ** 2)
    
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    return snr_db, signal_power, noise_power

# ==================== Data Processing Functions ====================

def swap_64_by_8(data):
    """Swap bytes within each 8-byte (64-bit) group"""
    arr = np.frombuffer(data, dtype=np.uint64)
    return arr.byteswap().tobytes()

def convert_real_imag(A):
    """Convert binary data to complex numbers (I + jQ)"""
    A = np.asarray(A, dtype=np.uint32)
    i_vals = (A & 0xffff).astype(np.int16)
    q_vals = ((A >> 16) & 0xffff).astype(np.int16)
    return i_vals.astype(np.float64) + 1j * q_vals.astype(np.float64)

def load_mbin_file(uploaded_file):
    """Load and parse .mbin file"""
    raw = uploaded_file.read()
    swapped = swap_64_by_8(raw)
    A = np.frombuffer(swapped, dtype=np.uint32)
    B = convert_real_imag(A)
    return B

# ==================== Streamlit App ====================

st.set_page_config(page_title="MMU Dump Plot", layout="wide")

st.title("📡 MMU Dump Plot - DMRS SNR Calculator")
st.markdown("### 3GPP TS 38.211 DMRS Analysis Tool")

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File upload
    uploaded_file = st.file_uploader("Upload .mbin File", type=['mbin'])
    
    st.divider()
    
    # DMRS Settings
    st.subheader("DMRS Configuration")
    
    pci = st.number_input("PCI", min_value=0, max_value=1007, value=1)
    dmrs_type = st.selectbox("DMRS Type", ["Type 1", "Type 2"])
    
    if dmrs_type == "Type 1":
        cdm_group = st.selectbox("CDM Group", [0, 1])
        generate_dmrs = generate_nr_dmrs_type1_3gpp
    else:
        cdm_group = st.selectbox("CDM Group", [0, 1, 2])
        generate_dmrs = generate_nr_dmrs_type2_3gpp
    
    slot = st.number_input("Slot", min_value=0, max_value=159, value=4)
    start_rb = st.number_input("Start RB", min_value=0, max_value=272, value=0)
    rb_size = st.number_input("RB Size", min_value=1, max_value=273, value=51)
    num_rb = st.number_input("NumRB (Total)", min_value=1, max_value=273, value=273)
    
    dmrs_syms_str = st.text_input("DMRS Symbols (comma separated)", "2, 11")
    dmrs_syms = [int(x.strip()) for x in dmrs_syms_str.split(',')]
    
    st.divider()
    
    # Action buttons
    run_analysis = st.button("🏃 Run Analysis", type="primary")

# Main content
if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("Loading file..."):
            raw_data = load_mbin_file(uploaded_file)
            
            TonePerRB = 12
            NumSym = 14
            NumTone = num_rb * TonePerRB
            
            SlotCnt = len(raw_data) // (NumTone * NumSym)
            C = raw_data[:SlotCnt * NumTone * NumSym].reshape(SlotCnt, NumTone * NumSym)
        
        st.success(f"✅ File loaded: {SlotCnt} slots, {NumTone} tones ({num_rb} RBs), {NumSym} symbols")
        
        if run_analysis:
            st.divider()
            st.header("📊 Analysis Results")
            
            # Extract slot data
            data = C[slot, :]
            dat_sym = data.reshape(NumSym, NumTone).T
            
            # Calculate SNR
            n_id = pci
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("DMRS SNR Results")
                snr_results = []
                
                for dmrs_sym in dmrs_syms:
                    rx_dmrs = dat_sym[:, dmrs_sym]
                    
                    ideal_dmrs, dmrs_indices = generate_dmrs(
                        n_rb=rb_size, start_rb=start_rb, slot=slot,
                        symbol_idx=dmrs_sym, n_id=n_id, cdm_group=cdm_group
                    )
                    snr_3gpp, sig_pwr, noise_pwr = calculate_dmrs_snr_3gpp(rx_dmrs, ideal_dmrs, dmrs_indices)
                    snr_results.append((dmrs_sym, snr_3gpp))
                    
                    st.metric(f"Symbol {dmrs_sym} SNR", f"{snr_3gpp:.2f} dB")
                
                # Channel estimation for EQ
                ideal_dmrs, dmrs_indices = generate_dmrs(
                    n_rb=rb_size, start_rb=start_rb, slot=slot,
                    symbol_idx=dmrs_syms[0], n_id=n_id, cdm_group=cdm_group
                )
                
                start_idx = start_rb * 12
                end_idx = (start_rb + rb_size) * 12
                rx_dmrs = dat_sym[dmrs_indices, dmrs_syms[0]]
                H_dmrs = rx_dmrs / ideal_dmrs
                
                H_full = np.ones(end_idx - start_idx, dtype=complex)
                for i, idx in enumerate(dmrs_indices):
                    local_idx = idx - start_idx
                    if 0 <= local_idx < len(H_full):
                        H_full[local_idx] = H_dmrs[i]
                
                for k in range(1, len(H_full), 2):
                    H_full[k] = H_full[k - 1]
                
                # EVM calculation
                data_symbols = [s for s in range(NumSym) if s not in dmrs_syms]
                eq_all = []
                for sym in data_symbols:
                    rx_data = dat_sym[start_idx:end_idx, sym]
                    eq_data = rx_data / H_full
                    eq_all.append(eq_data)
                
                eq_all = np.concatenate(eq_all)
                eq_normalized = eq_all / np.abs(eq_all)
                qpsk_points = (1 + 1j) / np.sqrt(2) * np.array([1+0j, 1j, -1+0j, -1j])
                evm = [np.min(np.abs(pt - qpsk_points)) ** 2 for pt in eq_normalized]
                evm_rms = np.sqrt(np.mean(evm)) * 100
                evm_db = 20 * np.log10(evm_rms / 100)
                
                st.metric("EVM (QPSK)", f"{evm_rms:.2f}% ({evm_db:.2f} dB)")
            
            with col2:
                st.subheader("EQ Constellation (Combined)")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(np.real(eq_all), np.imag(eq_all), marker='.', alpha=0.5, s=1)
                ax.set_xlabel('I')
                ax.set_ylabel('Q')
                ax.set_title(f'EQ Constellation (Slot {slot})')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                theta = np.linspace(0, 2*np.pi, 100)
                avg_amp = np.mean(np.abs(eq_all))
                ax.plot(avg_amp * np.cos(theta), avg_amp * np.sin(theta), 'r--', alpha=0.5)
                
                st.pyplot(fig)
            
            # Symbol-by-symbol EQ constellation
            st.subheader("EQ Constellation per Symbol")
            fig2, axes = plt.subplots(2, 7, figsize=(16, 5))
            fig2.suptitle(f'Slot {slot} - EQ Constellation per Symbol')
            
            eq_all_list = []
            for sym in range(NumSym):
                ax = axes[sym // 7, sym % 7]
                rx_data = dat_sym[start_idx:end_idx, sym]
                
                if sym in dmrs_syms:
                    ax.scatter(np.real(rx_data[::2]), np.imag(rx_data[::2]), marker='.', s=1)
                    ax.set_title(f'DMRS {sym}', fontsize=8, color='red')
                else:
                    eq_data = rx_data / H_full
                    eq_all_list.append(eq_data)
                    ax.scatter(np.real(eq_data), np.imag(eq_data), marker='.', s=1)
                    ax.set_title(f'Data {sym}', fontsize=8)
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')
            
            fig2.tight_layout()
            st.pyplot(fig2)
            
            # Additional plots
            st.divider()
            st.header("📈 Additional Plots")
            
            tab1, tab2, tab3 = st.tabs(["3D Waveform", "IQ per Symbol", "Magnitude per Symbol"])
            
            with tab1:
                st.subheader("3D Waveform")
                from mpl_toolkits.mplot3d import Axes3D
                
                fig3d = plt.figure(figsize=(10, 6))
                ax3d = fig3d.add_subplot(111, projection='3d')
                
                mag = np.abs(dat_sym)
                X, Y = np.meshgrid(np.arange(NumSym), np.arange(NumTone))
                ax3d.plot_surface(X, Y, mag, cmap='viridis', edgecolor='none')
                ax3d.set_xlabel('Time (Symbol)')
                ax3d.set_ylabel('Frequency (Tone)')
                ax3d.set_zlabel('Magnitude')
                ax3d.set_title(f'Slot {slot} - 3D Waveform')
                
                st.pyplot(fig3d)
            
            with tab2:
                st.subheader("IQ Constellation per Symbol (Raw)")
                fig_iq, axes_iq = plt.subplots(2, 7, figsize=(16, 5))
                fig_iq.suptitle(f'Slot {slot} - IQ Constellation per Symbol (Raw)')
                
                for sym in range(NumSym):
                    ax = axes_iq[sym // 7, sym % 7]
                    if sym in dmrs_syms:
                        ax.scatter(np.real(dat_sym[:, sym]), np.imag(dat_sym[:, sym]), marker='.')
                        ax.set_title(f'DMRS {sym}', fontsize=8, color='red')
                    else:
                        ax.scatter(np.real(dat_sym[::2, sym]), np.imag(dat_sym[::2, sym]), marker='.')
                        ax.set_title(f'Data {sym}', fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal')
                
                fig_iq.tight_layout()
                st.pyplot(fig_iq)
            
            with tab3:
                st.subheader("Magnitude per Symbol")
                fig_mag, axes_mag = plt.subplots(2, 7, figsize=(16, 5))
                fig_mag.suptitle(f'Slot {slot} - Magnitude per Symbol')
                
                for sym in range(NumSym):
                    ax = axes_mag[sym // 7, sym % 7]
                    mag_sym = np.abs(dat_sym[:, sym])
                    ax.plot(np.arange(len(mag_sym)), mag_sym)
                    if sym in dmrs_syms:
                        ax.set_title(f'DMRS {sym}', fontsize=8, color='red')
                    else:
                        ax.set_title(f'Data {sym}', fontsize=8)
                    ax.set_xlabel('Tone', fontsize=6)
                    ax.set_ylabel('Mag', fontsize=6)
                    ax.tick_params(axis='both', labelsize=5)
                
                fig_mag.tight_layout()
                st.pyplot(fig_mag)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("📁 Please upload a .mbin file to begin analysis.")
    
    st.markdown("""
    ### 📋 Instructions
    
    1. **Upload File**: Upload your `.mbin` file using the sidebar
    2. **Configure Settings**: Adjust DMRS parameters as needed
    3. **Run Analysis**: Click "Run Analysis" button
    
    ### 🔧 Features
    
    - **DMRS SNR Calculation**: 3GPP TS 38.211 compliant
    - **Channel Estimation**: Using DMRS sequence
    - **EQ Constellation**: Symbol-by-symbol equalization
    - **EVM Calculation**: QPSK modulation
    - **3D Waveform**: Time-frequency visualization
    - **DMRS Type 1 & 2**: Both supported
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    MMU Dump Plot - DMRS SNR Calculator | 3GPP TS 38.211
</div>
""", unsafe_allow_html=True)