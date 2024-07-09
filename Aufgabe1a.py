import numpy as np
import matplotlib.pyplot as plt

# Alpha- und Beta-Funktionen für die Gate-Variablen
def alpha_n(U): 
    return -0.01 * (55 + U) / (np.exp(-(55 + U) / 10) - 1)

def alpha_m(U): 
    return -0.1 * (40 + U) / (np.exp(-(40 + U) / 10) - 1)

def alpha_h(U): 
    return 0.07 * np.exp(-(65 + U) / 20)

def beta_n(U): 
    return 0.125 * np.exp(-(65 + U) / 80)

def beta_m(U): 
    return 4 * np.exp(-(65 + U) / 18)

def beta_h(U): 
    return 1 / (np.exp(-(35 + U) / 10) + 1)


# Berechnung der stationären Zustände der Gate-Variablen
def stat_Zustände_Gate(alpha, beta, U):
    return alpha(U) / (alpha(U) + beta(U))

# Parameter
C = 1.0  # Kapazität der Membran (µF/cm^2)
g_K = 36.0  # maximale Leitfähigkeit Kalium (mS/cm^2)
g_Na = 120.0  # maximale Leitfähigkeit Natrium (mS/cm^2)
g_L = 0.3  # Leitfähigkeit Leckstrom (mS/cm^2)
U_K = -77.0  # Gleichgewichtspotential Kalium (mV)
U_Na = 50.0  # Gleichgewichtspotential Natrium (mV)
U_L = -54.387  # Gleichgewichtspotential Leckstrom (mV)

# Zeitparameter
dt = 0.01  # Zeitschritt (ms)
t_max = 50  # maximale Zeit (ms)
time = np.arange(0, t_max + dt, dt)

# Anfangsbedingungen
U = -65.0  # Membranpotential (mV) bei geschlossenen Kanälen
I0 = 10.0  # konstante Stromstärke (µA/cm^2)

# Berechnung der Anfangswerte für m, h und n bei U = -65 mV
m0 = stat_Zustände_Gate(alpha_m, beta_m, U)
h0 = stat_Zustände_Gate(alpha_h, beta_h, U)
n0 = stat_Zustände_Gate(alpha_n, beta_n, U)

print(f'Werte der Gate-Variablen bei U = {U} mV:')
print(f'm = {m0}')
print(f'h = {h0}')
print(f'n = {n0}')

# Spannungsbereich
U_range = np.linspace(-100, 100, 500)

# Berechnung der stationären Zustände
m_inf = stat_Zustände_Gate(alpha_m, beta_m, U_range)
h_inf = stat_Zustände_Gate(alpha_h, beta_h, U_range)
n_inf = stat_Zustände_Gate(alpha_n, beta_n, U_range)

# Plotten der stationären Zustände der Gate-Variablen gegen Spannung
plt.figure(figsize=(10, 6))
plt.plot(U_range, m_inf, label='m_inf (Na Activation)')
plt.plot(U_range, h_inf, label='h_inf (Na Inactivation)')
plt.plot(U_range, n_inf, label='n_inf (K Activation)')

plt.xlabel('Membranpotential (mV)')
plt.ylabel('Gate-Variablen')
plt.title('Stationäre Zustände der Gate-Variablen der Hodgkin-Huxley-Gleichungen')
plt.legend()
plt.grid(True)
plt.show()

# Speicher für die Ergebnisse
U_Values = []
m_Values = []
h_Values = []
n_Values = []

# Anfangsbedingungen in den Speicher
U_Values.append(U)
m_Values.append(m0)
h_Values.append(h0)
n_Values.append(n0)

# Euler-Verfahren 
for t in time[1:]:
    # Ionische Ströme
    I_Na = g_Na * m_Values[-1]**3 * h_Values[-1] * (U - U_Na)
    I_K = g_K * n_Values[-1]**4 * (U - U_K)
    I_L = g_L * (U - U_L)
    
    # Differentialgleichungen
    dUdt = (I0 - I_Na - I_K - I_L) / C
    dmdt = alpha_m(U) * (1 - m_Values[-1]) - beta_m(U) * m_Values[-1]
    dhdt = alpha_h(U) * (1 - h_Values[-1]) - beta_h(U) * h_Values[-1]
    dndt = alpha_n(U) * (1 - n_Values[-1]) - beta_n(U) * n_Values[-1]
    
    # Update der Variablen
    U = U + dUdt * dt
    m_new = m_Values[-1] + dmdt * dt
    h_new = h_Values[-1] + dhdt * dt
    n_new = n_Values[-1] + dndt * dt
    
    # Speichern der neuen Werte
    U_Values.append(U)
    m_Values.append(m_new)
    h_Values.append(h_new)
    n_Values.append(n_new)

# Plot der zeitlichen Entwicklung des Membranpotentials und der Gate-Variablen
plt.figure(figsize=(12, 8))

# Membranpotential
plt.subplot(2, 1, 1)
plt.plot(time, U_Values, label='Membranpotential (U)')
plt.ylabel('Membranpotential (mU)')
plt.legend()

# Gate-Uariablen
plt.subplot(2, 1, 2)
plt.plot(time, m_Values, label='m')
plt.plot(time, h_Values, label='h')
plt.plot(time, n_Values, label='n')
plt.xlabel('Zeit (ms)')
plt.ylabel('Gate-Variablen')
plt.legend()

plt.tight_layout()
plt.show()

