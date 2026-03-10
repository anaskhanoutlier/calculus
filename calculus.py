

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint, quad, dblquad
from scipy.optimize import minimize, fsolve
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# SECTION 1: DIFFERENTIAL CALCULUS
# ─────────────────────────────────────────

def numerical_derivative(f, x, h=1e-5):
    """Central difference formula: f'(x) ≈ [f(x+h) - f(x-h)] / 2h"""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_second_derivative(f, x, h=1e-4):
    """f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²"""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2


def find_critical_points(f, x_range, h=1e-4):
    """
    Locate critical points where f'(x) ≈ 0.
    Uses sign change detection.
    """
    x_vals = np.linspace(*x_range, 5000)
    f_prime = np.array([numerical_derivative(f, xi) for xi in x_vals])
    
    critical = []
    for i in range(len(f_prime) - 1):
        if f_prime[i] * f_prime[i+1] < 0:
            xc = (x_vals[i] + x_vals[i+1]) / 2
            f2 = numerical_second_derivative(f, xc)
            point_type = "Local Min" if f2 > 0 else ("Local Max" if f2 < 0 else "Inflection")
            critical.append((xc, f(xc), point_type))
    return critical


# Test functions
f_poly   = lambda x: x**4 - 4*x**3 + 4*x**2 + 1
f_trig   = lambda x: np.sin(x) * np.exp(-0.2*x)
f_combo  = lambda x: x**3 - 3*x + 1


# ─────────────────────────────────────────
# SECTION 2: INTEGRAL CALCULUS
# ─────────────────────────────────────────

def riemann_sum(f, a, b, n, method='midpoint'):
    """
    Compute Riemann sum approximation of ∫f(x)dx from a to b.
    Methods: left, right, midpoint, trapezoidal, simpson
    """
    h = (b - a) / n
    if method == 'left':
        x = np.linspace(a, b - h, n)
        return h * np.sum(f(x))
    elif method == 'right':
        x = np.linspace(a + h, b, n)
        return h * np.sum(f(x))
    elif method == 'midpoint':
        x = np.linspace(a + h/2, b - h/2, n)
        return h * np.sum(f(x))
    elif method == 'trapezoidal':
        x = np.linspace(a, b, n + 1)
        return h * (f(x[0])/2 + np.sum(f(x[1:-1])) + f(x[-1])/2)
    elif method == 'simpson':
        if n % 2 != 0:
            n += 1
        x = np.linspace(a, b, n + 1)
        return (h/3) * (f(x[0]) + 4*np.sum(f(x[1::2])) + 2*np.sum(f(x[2:-1:2])) + f(x[-1]))


def area_between_curves(f1, f2, a, b, n=10000):
    """Area between two curves: ∫|f1(x) - f2(x)|dx"""
    x = np.linspace(a, b, n)
    return np.trapz(np.abs(f1(x) - f2(x)), x)


# ─────────────────────────────────────────
# SECTION 3: TAYLOR SERIES
# ─────────────────────────────────────────

def taylor_series_sin(x, n_terms):
    """
    Taylor series for sin(x) around x=0:
    sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
    """
    from math import factorial
    result = np.zeros_like(x, dtype=float)
    for k in range(n_terms):
        n = 2*k + 1
        result += ((-1)**k / factorial(n)) * x**n
    return result


def taylor_series_exp(x, n_terms):
    """
    Taylor series for e^x around x=0:
    e^x = 1 + x + x²/2! + x³/3! + ...
    """
    from math import factorial
    result = np.zeros_like(x, dtype=float)
    for k in range(n_terms):
        result += x**k / factorial(k)
    return result


def taylor_series_ln(x, n_terms):
    """
    Taylor series for ln(1+x) around x=0 (|x|<1):
    ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
    """
    result = np.zeros_like(x, dtype=float)
    for k in range(1, n_terms + 1):
        result += ((-1)**(k+1) / k) * x**k
    return result


# ─────────────────────────────────────────
# SECTION 4: FOURIER SERIES
# ─────────────────────────────────────────

def fourier_series_square_wave(x, n_terms):
    """
    Fourier series approximation of a square wave f(x) = sign(sin(x)):
    f(x) = (4/π) Σ sin((2k-1)x) / (2k-1)   k=1,2,3,...
    """
    result = np.zeros_like(x, dtype=float)
    for k in range(1, n_terms + 1):
        n = 2*k - 1
        result += np.sin(n * x) / n
    return (4 / np.pi) * result


def fourier_series_sawtooth(x, n_terms):
    """
    Fourier series for sawtooth wave:
    f(x) = -2/π Σ (-1)^k sin(kx) / k   k=1,2,...
    """
    result = np.zeros_like(x, dtype=float)
    for k in range(1, n_terms + 1):
        result += ((-1)**k) * np.sin(k * x) / k
    return (-2 / np.pi) * result


# ─────────────────────────────────────────
# SECTION 5: ODE SYSTEMS
# ─────────────────────────────────────────

def lotka_volterra(state, t, alpha, beta, delta, gamma):
    """
    Lotka-Volterra Predator-Prey Model:
    dx/dt = αx - βxy        (prey grows, dies from predation)
    dy/dt = δxy - γy        (predator grows from prey, dies naturally)
    """
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]


def sir_model(state, t, beta, gamma):
    """
    SIR Epidemic Model:
    dS/dt = -β·S·I/N
    dI/dt = β·S·I/N - γ·I
    dR/dt = γ·I
    """
    S, I, R = state
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt =  beta * S * I / N - gamma * I
    dRdt =  gamma * I
    return [dSdt, dIdt, dRdt]


def damped_oscillator(state, t, omega0, zeta):
    """
    Damped Harmonic Oscillator:
    x'' + 2ζω₀x' + ω₀²x = 0
    Rewritten as system: x'=v, v'=-ω₀²x - 2ζω₀v
    """
    x, v = state
    dxdt = v
    dvdt = -omega0**2 * x - 2*zeta*omega0 * v
    return [dxdt, dvdt]


# ─────────────────────────────────────────
# SECTION 6: MULTIVARIABLE CALCULUS
# ─────────────────────────────────────────

def gradient_2d(f, x, y, h=1e-5):
    """Numerical gradient ∇f = (∂f/∂x, ∂f/∂y)"""
    df_dx = (f(x + h, y) - f(x - h, y)) / (2*h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2*h)
    return df_dx, df_dy


# 2D functions for surface plots
f_saddle  = lambda x, y: x**2 - y**2
f_bowl    = lambda x, y: x**2 + y**2
f_rosenbrock = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2  # classic optimization
f_wave    = lambda x, y: np.sin(np.sqrt(x**2 + y**2)) / (np.sqrt(x**2 + y**2) + 1e-8)


# ─────────────────────────────────────────
# SECTION 7: VISUALIZATION
# ─────────────────────────────────────────

def visualize_all():
    sns_style_applied = False
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns_style_applied = True
    except ImportError:
        pass

    # ── Figure 1: Calculus Overview ──
    fig1 = plt.figure(figsize=(18, 14))
    fig1.suptitle("Calculus & Differential Equations — Mathematical Analysis\n"
                  "BSc Mathematics | IGNOU | Python · NumPy · SciPy · Matplotlib",
                  fontsize=13, fontweight='bold')
    gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.52, wspace=0.38)

    # ── Plot 1: Function + Derivative + Critical Points ──
    ax = fig1.add_subplot(gs1[0, 0])
    x_plot = np.linspace(-1, 4, 500)
    y_vals = f_poly(x_plot)
    dy_vals = np.array([numerical_derivative(f_poly, xi) for xi in x_plot])
    ax.plot(x_plot, y_vals, 'b-', linewidth=2.5, label='f(x) = x⁴−4x³+4x²+1')
    ax.plot(x_plot, dy_vals, 'r--', linewidth=1.5, label="f'(x)")
    ax.axhline(0, color='black', linewidth=0.8)
    crits = find_critical_points(f_poly, (-1, 4))
    for xc, yc, ctype in crits:
        color = 'green' if 'Min' in ctype else 'red'
        ax.scatter([xc], [f_poly(xc)], color=color, s=80, zorder=5)
        ax.annotate(ctype, (xc, f_poly(xc)), fontsize=6.5, ha='center',
                    xytext=(0, 10), textcoords='offset points')
    ax.set_title("Differentiation &\nCritical Points", fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 6)

    # ── Plot 2: Riemann Sum Convergence ──
    ax = fig1.add_subplot(gs1[0, 1])
    f_int = lambda x: np.sin(x) * np.exp(-0.1*x)
    true_val, _ = quad(f_int, 0, np.pi)
    n_vals = [5, 10, 20, 50, 100, 200, 500, 1000]
    for method, color, style in [('left', 'blue', 'o-'),
                                   ('midpoint', 'green', 's-'),
                                   ('simpson', 'red', '^-')]:
        errors = [abs(riemann_sum(f_int, 0, np.pi, n, method) - true_val) for n in n_vals]
        ax.loglog(n_vals, errors, style, color=color, linewidth=1.5,
                  markersize=5, label=method.capitalize())
    ax.set_xlabel("Number of intervals n")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Integration Error Convergence\n(Log-Log Scale)", fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    # ── Plot 3: Taylor Series Approximation (sin) ──
    ax = fig1.add_subplot(gs1[0, 2])
    x_t = np.linspace(-2*np.pi, 2*np.pi, 500)
    ax.plot(x_t, np.sin(x_t), 'k-', linewidth=3, label='sin(x) (exact)', zorder=5)
    for n, color in [(1, 'red'), (3, 'orange'), (5, 'green'), (9, 'blue')]:
        y_approx = taylor_series_sin(x_t, n)
        ax.plot(x_t, y_approx, '--', color=color, linewidth=1.5,
                label=f'{n} terms')
    ax.set_ylim(-3, 3)
    ax.set_title("Taylor Series: sin(x)\nAround x=0", fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Fourier Series (Square Wave) ──
    ax = fig1.add_subplot(gs1[1, 0])
    x_f = np.linspace(-np.pi, np.pi, 1000)
    square = np.sign(np.sin(x_f))
    ax.plot(x_f, square, 'k-', linewidth=2, label='Square wave', alpha=0.5, zorder=5)
    for n_terms, color in [(1, 'red'), (3, 'orange'), (5, 'green'), (15, 'blue')]:
        ax.plot(x_f, fourier_series_square_wave(x_f, n_terms), '--',
                color=color, linewidth=1.5, label=f'{n_terms} harmonics')
    ax.set_title("Fourier Series\nSquare Wave Approximation", fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-np.pi, np.pi)

    # ── Plot 5: Fourier Series (Sawtooth) ──
    ax = fig1.add_subplot(gs1[1, 1])
    x_f2 = np.linspace(-np.pi+0.01, np.pi-0.01, 1000)
    true_saw = x_f2 / np.pi
    ax.plot(x_f2, true_saw, 'k-', linewidth=2, label='Sawtooth', alpha=0.5, zorder=5)
    for n_terms, color in [(2, 'red'), (5, 'orange'), (10, 'green'), (30, 'blue')]:
        ax.plot(x_f2, fourier_series_sawtooth(x_f2, n_terms), '--',
                color=color, linewidth=1.5, label=f'{n_terms} harmonics')
    ax.set_title("Fourier Series\nSawtooth Wave Approximation", fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)

    # ── Plot 6: Lotka-Volterra (Predator-Prey ODE) ──
    ax = fig1.add_subplot(gs1[1, 2])
    t_lv = np.linspace(0, 40, 2000)
    alpha_lv, beta_lv, delta_lv, gamma_lv = 0.6, 0.05, 0.025, 0.4
    sol_lv = odeint(lotka_volterra, [40, 5], t_lv,
                    args=(alpha_lv, beta_lv, delta_lv, gamma_lv))
    ax.plot(t_lv, sol_lv[:, 0], 'b-', linewidth=2, label='Prey x(t)')
    ax.plot(t_lv, sol_lv[:, 1], 'r-', linewidth=2, label='Predator y(t)')
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title("Lotka-Volterra ODE System\nPredator-Prey Model", fontweight='bold', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 7: SIR Model ──
    ax = fig1.add_subplot(gs1[2, 0])
    t_sir = np.linspace(0, 160, 1000)
    N = 10000
    sol_sir = odeint(sir_model, [N-10, 10, 0], t_sir, args=(0.3, 0.05))
    ax.plot(t_sir, sol_sir[:, 0], 'b-', linewidth=2, label='Susceptible S(t)')
    ax.plot(t_sir, sol_sir[:, 1], 'r-', linewidth=2, label='Infected I(t)')
    ax.plot(t_sir, sol_sir[:, 2], 'g-', linewidth=2, label='Recovered R(t)')
    ax.set_xlabel("Days")
    ax.set_ylabel("Population")
    ax.set_title("SIR Epidemic Model\n(β=0.3, γ=0.05, N=10000)", fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Plot 8: Phase Portrait (Lotka-Volterra) ──
    ax = fig1.add_subplot(gs1[2, 1])
    ax.plot(sol_lv[:, 0], sol_lv[:, 1], 'purple', linewidth=1.5)
    ax.scatter([sol_lv[0, 0]], [sol_lv[0, 1]], color='green', s=80, zorder=5, label='Start')
    ax.set_xlabel("Prey Population")
    ax.set_ylabel("Predator Population")
    ax.set_title("Phase Portrait\nLotka-Volterra", fontweight='bold', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 9: Damped Oscillator ──
    ax = fig1.add_subplot(gs1[2, 2])
    t_osc = np.linspace(0, 20, 1000)
    omega0 = 2.0
    for zeta, color, label in [(0.0, 'blue',   'Undamped'),
                                (0.2, 'green',  'Underdamped'),
                                (1.0, 'orange', 'Critically Damped'),
                                (2.0, 'red',    'Overdamped')]:
        sol = odeint(damped_oscillator, [1, 0], t_osc, args=(omega0, zeta))
        ax.plot(t_osc, sol[:, 0], color=color, linewidth=2, label=f'ζ={zeta}')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement x(t)")
    ax.set_title("Damped Oscillator ODE\nx''+2ζω₀x'+ω₀²x=0", fontweight='bold', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)

    plt.savefig("project13_calculus_odes.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ── Figure 2: 3D Surfaces (Multivariable Calculus) ──
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle("Multivariable Calculus — 3D Surface Plots & Gradient Fields\n"
                  "BSc Mathematics | Python · NumPy · Matplotlib 3D",
                  fontsize=12, fontweight='bold')

    x_3d = np.linspace(-3, 3, 80)
    y_3d = np.linspace(-3, 3, 80)
    X, Y = np.meshgrid(x_3d, y_3d)

    surface_configs = [
        (f_bowl,     "Bowl: f=x²+y²\n(Local Minimum)", 'Blues'),
        (f_saddle,   "Saddle: f=x²−y²\n(Saddle Point)", 'RdBu'),
        (f_wave,     "Wave: sin(√(x²+y²))/r\n(Radial)", 'viridis'),
        (f_rosenbrock, "Rosenbrock Function\n(Optimization Benchmark)", 'plasma'),
    ]

    for i, (func, title, cmap) in enumerate(surface_configs, 1):
        ax = fig2.add_subplot(2, 2, i, projection='3d')
        Z = func(X, Y)
        Z_clipped = np.clip(Z, np.percentile(Z, 2), np.percentile(Z, 98))
        surf = ax.plot_surface(X, Y, Z_clipped, cmap=cmap, alpha=0.85,
                               linewidth=0, antialiased=True)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("f(x,y)", fontsize=8)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.tick_params(labelsize=6)
        fig2.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig("project13_3d_surfaces.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Plots: project13_calculus_odes.png, project13_3d_surfaces.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 65)
    print("PROJECT 13: Calculus & Differential Equations")
    print("BSc Mathematics (Major) | IGNOU")
    print("Python · NumPy · SciPy · Matplotlib")
    print("=" * 65)

    # Numerical integration comparison
    f_int = lambda x: np.sin(x)
    true_val = 2.0  # ∫sin(x)dx from 0 to π
    print("\n━━━ NUMERICAL INTEGRATION: ∫sin(x)dx from 0 to π ━━━")
    for method in ['left', 'right', 'midpoint', 'trapezoidal', 'simpson']:
        approx = riemann_sum(f_int, 0, np.pi, 1000, method)
        print(f"  {method:<15}: {approx:.8f}  (error = {abs(approx-true_val):.2e})")

    # Double integral (SciPy)
    f_double = lambda y, x: x**2 * y
    result, error = dblquad(f_double, 0, 2, 0, 3)
    print(f"\n  Double Integral ∫∫ x²y dydx (x:0→2, y:0→3) = {result:.6f}  (exact = 18)")

    # Critical point analysis
    print("\n━━━ CRITICAL POINTS: f(x) = x⁴−4x³+4x²+1 ━━━")
    crits = find_critical_points(f_poly, (-0.5, 3.5))
    for xc, yc, ctype in crits:
        print(f"  x = {xc:.4f},  f(x) = {yc:.4f}  →  {ctype}")

    # Taylor series error
    print("\n━━━ TAYLOR SERIES ERROR: sin(π/4) ━━━")
    x0 = np.pi / 4
    true_sin = np.sin(x0)
    for n in [1, 2, 3, 5, 7, 10]:
        approx = taylor_series_sin(np.array([x0]), n)[0]
        print(f"  n={n:>2} terms: {approx:.8f}  error={abs(approx-true_sin):.2e}")

    # Lotka-Volterra equilibrium
    alpha, beta, delta, gamma = 0.6, 0.05, 0.025, 0.4
    x_eq = gamma / delta
    y_eq = alpha / beta
    print(f"\n━━━ LOTKA-VOLTERRA EQUILIBRIUM ━━━")
    print(f"  Prey equilibrium    : x* = γ/δ = {x_eq:.2f}")
    print(f"  Predator equilibrium: y* = α/β = {y_eq:.2f}")

    # SIR basic reproduction number
    beta_sir, gamma_sir, N = 0.3, 0.05, 10000
    R0 = beta_sir / gamma_sir
    print(f"\n━━━ SIR MODEL: Basic Reproduction Number ━━━")
    print(f"  R₀ = β/γ = {R0:.2f}  → {'Epidemic spreads ⚠' if R0 > 1 else 'No epidemic'}")

    print("\n📊 Generating visualizations...")
    visualize_all()


if __name__ == "__main__":
    main()
