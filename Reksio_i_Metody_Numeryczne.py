import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
import plotly.graph_objects as go

def natural_cubic_spline(x_nodes, y_nodes):
    """
    Oblicza współczynniki naturalnych splajnów sześciennych
    Zwraca: a, b, c, d - współczynniki wielomianów sześciennych
    """
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)

    # Obliczanie współczynników alpha
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (y_nodes[i + 1] - y_nodes[i]) - (3 / h[i - 1]) * (y_nodes[i] - y_nodes[i - 1])

    # Rozwiązanie układu równań dla c
    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)

    for i in range(1, n):
        l[i] = 2 * (x_nodes[i + 1] - x_nodes[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    # Wsteczna substytucja
    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_nodes[j + 1] - y_nodes[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    a = y_nodes[:-1]
    return a, b, c[:-1], d


def create_spline_function(x_nodes, y_nodes, method='natural'):
    """
    Tworzy funkcję interpolującą na podstawie wybranej metody
    """
    if method == 'natural':
        a, b, c, d = natural_cubic_spline(x_nodes, y_nodes)

        def spline_function(x):
            x = np.asarray(x) if isinstance(x, (list, tuple, np.ndarray)) else np.array([x])
            y = np.zeros_like(x, dtype=float)

            for i in range(len(x_nodes) - 1):
                mask = (x >= x_nodes[i]) & (x <= x_nodes[i + 1])
                if np.any(mask):
                    dx = x[mask] - x_nodes[i]
                    y[mask] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

            return y[0] if y.size == 1 else y

        return spline_function

    elif method == 'pchip':
        pchip = PchipInterpolator(x_nodes, y_nodes)
        return pchip

    elif method == 'fit':
        x_fit = []
        y_fit = []
        weights = []

        for (x_min, x_max), y in zip(x_ranges, y_nodes):
            if x_min == x_max:
                x_fit.append(x_min)
                y_fit.append(y)
                weights.append(1.0)
            else:
                x_fit.extend([x_min, x_max])
                y_fit.extend([y, y])
                weights.extend([0.5, 0.5])

        sigma = 1 / np.sqrt(weights)

        def exp_model(x, a, b, c):
            return a * np.exp(b * x) + c

        params, _ = curve_fit(
            exp_model, x_fit, y_fit,
            p0=[1, 0.0001, 0],
            sigma=sigma,
            maxfev=10000,
            bounds=([-np.inf, -0.1, -np.inf], [np.inf, 0.1, np.inf]))

        a, b, c = params
        return lambda x: exp_model(x, a, b, c)

    else:
        raise ValueError("Dostępne metody: 'natural' lub 'pchip'")


def plot_spline(x_nodes, y_nodes, spline_func, title):
    """
    Creates an interactive spline interpolation plot using Plotly.
    """
    x_fine = np.linspace(min(x_nodes), max(x_nodes), abs(min(x_nodes)) + abs(max(x_nodes)))
    y_fine = spline_func(x_fine)

    # Convert minutes to HH:MM AM/PM format
    def min_to_hhmm(minutes):
        hh = int(minutes) // 60
        mm = int(minutes) % 60
        period = "AM" if hh < 12 else "PM"
        hh = hh % 12
        hh = 12 if hh == 0 and period == "PM" else hh  # Handle 0 as 12 AM/PM
        return f"{hh:02d}:{mm:02d} {period}"

    # Create the figure
    fig = go.Figure()

    # Add spline curve
    fig.add_trace(go.Scatter(
        x=x_fine,
        y=y_fine,
        mode='lines',
        name='Interpolacja',
        line=dict(color='blue'),
        hoverinfo='none'
    ))
    # Add invisible trace for hover
    fig.add_trace(go.Scatter(
        x=x_fine,
        y=y_fine,
        mode='none',
        hoverinfo='text',
        hovertext=[f"({int(x)}, {minutes_to_hhmmss_ampm(y)})" for x, y in zip(x_fine, y_fine)],
        showlegend=False
    ))

    # RED CIRCLES
    fig.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers',
        name='daty RiWC',
        marker=dict(color='red', size=10),
        hoverinfo='none'
    ))
    date_names = ['Prehistoria', 'Egipt', 'Średniowiecze', 'Wiek XIX', 'Przyszłość']
    i = 0
    for xi, yi in zip(x_nodes, y_nodes):
        fig.add_trace(go.Scatter(
            x=[xi],
            y=[yi],
            mode='markers+text',
            name=date_names[i],
            marker=dict(color='rgba(0,0,0,0)', size=10),
            text=[f"({xi}, {min_to_hhmm(yi)})"],
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hoverinfo='none'
        ))
        i += 1

    # Customize layout
    fig.update_layout(
        title={
            'text': f'RiWC Flow of Time ({title})',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20)
        },
        xaxis_title='Rok [w rokach]',
        yaxis_title='Czas na zegarze',
        yaxis_range=[0, 1440],
        hovermode='x',
        template='plotly_white',
        updatemenus=[{
            'type': 'buttons',
            'direction': 'left',
            'x': 0.0,
            'y': -0.25,
            'buttons': [
                {
                    'label': 'Linear Scale',
                    'method': 'relayout',
                    'args': [{'xaxis.type': 'linear', 'xaxis.title.text': 'Rok [w rokach]'}]
                },
                {
                    'label': 'Log Scale',
                    'method': 'relayout',
                    'args': [{'xaxis.type': 'log', 'xaxis.title.text': 'Rok [w rokach]'}]
                }
            ]
        }]
    )
    # Set initial axis to log (default)
    fig.update_xaxes(type='linear')

    # Custom y-axis tick formatting (minutes → HH:MM AM/PM)
    fig.update_yaxes(
        tickvals=np.linspace(0, 1440, 13),  # Every 2 hours
        ticktext=[min_to_hhmm(x) for x in np.linspace(0, 1440, 13)]
    )

    # Save as interactive HTML
    fig.write_html(f"{title}.html")
    print(f"Interactive plot saved as: {title}.html")
    if title == "PCHIP":
        fig.write_html(f"index.html")

    # Show in notebook (optional)
    fig.show()

def interpolate_with_splines(x, y, method='natural'):
    """
    Główna funkcja wykonująca interpolację wybraną metodą
    """
    x_nodes = np.array(x)
    y_nodes = np.array(y)

    # Sortowanie węzłów (ważne dla PCHIP)
    sort_idx = np.argsort(x_nodes)
    x_nodes = x_nodes[sort_idx]
    y_nodes = y_nodes[sort_idx]



    # Tworzenie funkcji interpolującej
    spline_func = create_spline_function(x_nodes, y_nodes, method)

    # Rysowanie wykresu
    method_name = 'Splajn naturalny' if method == 'natural' else 'PCHIP' if method == "pchip" else 'Fit curve'
    plot_spline(x_nodes, y_nodes, spline_func, method_name)

    return spline_func

def minutes_to_hhmmss_ampm(minutes):
    """Convert minutes to hh:mm:ss AM/PM format"""
    total_seconds = int(minutes * 60)
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    period = 'AM' if hours < 12 else 'PM'
    display_hours = hours if hours <= 12 else hours - 12
    if display_hours == 0 and period == 'PM':  # Handle midnight
        display_hours = 12

    return f"{display_hours:02d}:{minutes:02d}:{seconds:02d} {period}"

# Przykładowe dane
x = [-40000, -2650, 1500, 1870, 2050]
y = [180, 9*60+34, 15*60, 19*60+30, 20*60+30]

x_ranges = [
    (-40000, -30000),
    (-2550, -2500),
    (1503, 1506),
    (1870, 1890),
    (2049, 2050)
]

# Wykonanie interpolacji naturalnym splajnem
# print("Naturalny splajn sześcienny:")
# spline_natural = interpolate_with_splines(x, y, method='natural')
#
# natural_value = spline_natural(350)
# print(f"Wartość splajna w x=350: {minutes_to_hhmmss_ampm(natural_value)}")

# Wykonanie interpolacji PCHIP
print("\nPCHIP (monotoniczna interpolacja):")
spline_pchip = interpolate_with_splines(x, y, method='pchip')

interpolated_value = spline_pchip(350)
print(f"Wartość PCHIP w x=350: {minutes_to_hhmmss_ampm(interpolated_value)}")