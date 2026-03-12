"""
Behavioral Matching Engine — MVP Validation
============================================
Validates all four mathematical pillars using fully synthetic data.
Outputs a multi-panel matplotlib figure showing score distributions
and behavioral dynamics.

Pillars validated:
  1. Dynamic Profile Vector (time-decayed, per-category lambda)
  2. Cosine Similarity vs. Naive Preference Matching
  3. Adaptive Weight Schedule (n_eff = min(n_A, n_B))
  4. Venue Recommendation Scoring (unit-normalized, hard gate)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import cosine
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Global Config ─────────────────────────────────────────────────────────────
CATEGORIES = ["Gym", "Coffee", "Bar", "Park", "Restaurant", "Library", "Nightclub", "Retail"]
N_CATS = len(CATEGORIES)

# Per-category decay rates (lambda per day) — slower = stickier habit
LAMBDA = np.array([0.02, 0.05, 0.08, 0.01, 0.07, 0.01, 0.10, 0.06])

# IDF weights (simulated — in production, computed from user base)
# Lower df = rarer category = higher IDF boost
DF = np.array([300, 800, 600, 400, 900, 150, 200, 700])  # users per category
N_USERS_TOTAL = 1000
IDF = np.log(N_USERS_TOTAL / DF)

# Adaptive weight params
W2_FIXED = 0.15       # proximity weight (fixed)
W3_MAX   = 0.60       # bio weight at zero pings
MU       = 0.03       # decay rate for bio weight

# Venue recommendation weights
ALPHA, BETA, GAMMA = 0.70, 0.10, 0.20


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — Dynamic Profile Vector
# ═══════════════════════════════════════════════════════════════════════════════

def make_ping(category_idx, dwell_minutes):
    """Single ping: weighted unit vector in R^n."""
    w = np.log10(1 + dwell_minutes / 30)
    vec = np.zeros(N_CATS)
    vec[category_idx] = w
    return vec

def decay_vector(V, delta_t):
    """Apply per-category exponential decay over delta_t days."""
    return V * np.exp(-LAMBDA * delta_t)

def build_profile(ping_schedule):
    """
    Build a lifestyle vector from a ping schedule.
    ping_schedule: list of (delta_t_days, category_idx, dwell_minutes)
    Returns V (raw) and history of (n_pings, V_snapshot)
    """
    V = np.zeros(N_CATS)
    history = []
    n_pings = 0
    for delta_t, cat, dwell in ping_schedule:
        V = decay_vector(V, delta_t)
        V += make_ping(cat, dwell)
        n_pings += 1
        history.append((n_pings, V.copy()))
    return V, history

def idf_weight(V):
    """Apply IDF weighting and unit-normalize."""
    V_star = V * IDF
    norm = np.linalg.norm(V_star)
    if norm == 0:
        return V_star
    return V_star / norm


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 — Similarity: Behavioral (Cosine) vs. Naive (Preference Overlap)
# ═══════════════════════════════════════════════════════════════════════════════

def behavioral_similarity(V_A, V_B):
    """Cosine similarity on IDF-weighted, unit-normalized vectors."""
    A = idf_weight(V_A)
    B = idf_weight(V_B)
    if np.linalg.norm(A) == 0 or np.linalg.norm(B) == 0:
        return 0.0
    return float(np.dot(A, B))  # already unit-normalized, so dot = cosine sim

def naive_similarity(tags_A, tags_B):
    """Naive preference matching: Jaccard overlap of stated interest tags."""
    set_A, set_B = set(tags_A), set(tags_B)
    if not set_A and not set_B:
        return 0.0
    return len(set_A & set_B) / len(set_A | set_B)


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 — Gaussian Proximity Penalty
# ═══════════════════════════════════════════════════════════════════════════════

def proximity_score(commute_minutes, sigma=25):
    """RBF kernel on commute time. sigma tuned per city."""
    return np.exp(-(commute_minutes**2) / (2 * sigma**2))


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 4 — Adaptive Weight + Holistic Match Score
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_weights(n_A, n_B):
    """Compute (w1, w2, w3) using n_eff = min(n_A, n_B)."""
    n_eff = min(n_A, n_B)
    w3 = W3_MAX * np.exp(-MU * n_eff)
    w1 = 1 - W2_FIXED - w3
    w1 = max(w1, 0)
    return w1, W2_FIXED, w3

def holistic_score(V_A, V_B, n_A, n_B, tags_A, tags_B, commute_minutes, sigma=25):
    """Global match score G(A, B)."""
    w1, w2, w3 = adaptive_weights(n_A, n_B)
    sim_beh  = behavioral_similarity(V_A, V_B)
    sim_log  = proximity_score(commute_minutes, sigma)
    sim_prior = naive_similarity(tags_A, tags_B)
    return w1 * sim_beh + w2 * sim_log + w3 * sim_prior


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 5 — Venue Recommendation Score
# ═══════════════════════════════════════════════════════════════════════════════

VENUES = [
    {"name": "Iron & Oak Gym",      "category": 0, "partner": False, "travel_A": 8,  "travel_B": 12},
    {"name": "Monarch Coffee",      "category": 1, "partner": True,  "travel_A": 5,  "travel_B": 7},
    {"name": "The Rooftop Bar",     "category": 2, "partner": True,  "travel_A": 15, "travel_B": 10},
    {"name": "Loose Park",          "category": 3, "partner": False, "travel_A": 10, "travel_B": 8},
    {"name": "Corvino Restaurant",  "category": 4, "partner": True,  "travel_A": 12, "travel_B": 14},
    {"name": "KC Public Library",   "category": 5, "partner": False, "travel_A": 20, "travel_B": 18},
    {"name": "Midwest Coffee Co.",  "category": 1, "partner": False, "travel_A": 6,  "travel_B": 9},
    {"name": "Power & Light Bar",   "category": 6, "partner": True,  "travel_A": 18, "travel_B": 20},
]

THETA = 0.25  # hard gate threshold (meaningful with unit-normalized vectors)
P_MAX = 60    # max commute minutes for normalization

def venue_score(V_A, V_B, venue):
    """Score a venue for a matched pair."""
    A_norm = idf_weight(V_A)
    B_norm = idf_weight(V_B)
    c = venue["category"]
    intersection = A_norm[c] * B_norm[c]

    # Hard gate
    if intersection < THETA:
        return None  # excluded

    incentive = 1.0 if venue["partner"] else 0.0
    travel_avg = (venue["travel_A"] + venue["travel_B"]) / 2
    p_travel = (travel_avg / P_MAX)

    score = ALPHA * intersection + BETA * incentive - GAMMA * p_travel
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC USER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def make_user(archetype):
    """
    Generate a synthetic user with a ping schedule reflecting an archetype.
    Returns (ping_schedule, tags, commute_minutes_to_city_center)
    archetypes: 'fitness', 'bookworm', 'social', 'homebody', 'foodie'
    """
    archetypes = {
        # (category_weights, dwell_mean, dwell_std, n_pings)
        "fitness":   ([0.50, 0.15, 0.05, 0.15, 0.10, 0.02, 0.01, 0.02], 55, 15, 60),
        "bookworm":  ([0.05, 0.25, 0.02, 0.20, 0.10, 0.30, 0.02, 0.06], 45, 20, 45),
        "social":    ([0.05, 0.15, 0.30, 0.10, 0.20, 0.02, 0.15, 0.03], 50, 25, 50),
        "homebody":  ([0.10, 0.20, 0.05, 0.30, 0.15, 0.10, 0.02, 0.08], 30, 10, 20),
        "foodie":    ([0.05, 0.10, 0.10, 0.10, 0.50, 0.05, 0.05, 0.05], 70, 20, 55),
    }
    tag_map = {
        "fitness":  ["hiking", "yoga", "meal prep", "outdoors"],
        "bookworm": ["reading", "coffee", "museums", "hiking"],
        "social":   ["live music", "travel", "nightlife", "coffee"],
        "homebody": ["cooking", "movies", "hiking", "board games"],
        "foodie":   ["cooking", "wine", "travel", "markets"],
    }

    weights, dwell_mean, dwell_std, n_pings = archetypes[archetype]
    schedule = []
    for _ in range(n_pings):
        delta_t = np.random.exponential(0.5)  # avg ping every 12 hours
        cat = np.random.choice(N_CATS, p=weights)
        dwell = max(5, np.random.normal(dwell_mean, dwell_std))
        schedule.append((delta_t, cat, dwell))

    commute = np.random.randint(5, 35)
    return schedule, tag_map[archetype], commute


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TEST POPULATION
# ═══════════════════════════════════════════════════════════════════════════════

ARCHETYPES = ["fitness", "bookworm", "social", "homebody", "foodie"]
N_PER_ARCH = 10

users = []
for arch in ARCHETYPES:
    for _ in range(N_PER_ARCH):
        schedule, tags, commute = make_user(arch)
        V, history = build_profile(schedule)
        users.append({
            "archetype": arch,
            "V": V,
            "tags": tags,
            "n_pings": len(schedule),
            "history": history,
            "commute": commute,
        })

# Focus user: a fitness-archetype user (index 0)
focus = users[0]


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL MATCH SCORES AGAINST FOCUS USER
# ═══════════════════════════════════════════════════════════════════════════════

results = []
for i, other in enumerate(users[1:], 1):
    avg_commute = (focus["commute"] + other["commute"]) / 2
    beh  = behavioral_similarity(focus["V"], other["V"])
    naive = naive_similarity(focus["tags"], other["tags"])
    g    = holistic_score(
        focus["V"], other["V"],
        focus["n_pings"], other["n_pings"],
        focus["tags"], other["tags"],
        avg_commute
    )
    results.append({
        "archetype": other["archetype"],
        "behavioral": beh,
        "naive": naive,
        "holistic": g,
        "n_pings_other": other["n_pings"],
    })

arch_colors = {
    "fitness":  "#2196F3",
    "bookworm": "#9C27B0",
    "social":   "#FF9800",
    "homebody": "#4CAF50",
    "foodie":   "#F44336",
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 14), facecolor="#0F1923")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                       left=0.06, right=0.97, top=0.91, bottom=0.06)

TITLE_COLOR  = "#E8EDF2"
LABEL_COLOR  = "#A8B8C8"
GRID_COLOR   = "#1E2D3D"
PANEL_BG     = "#152030"
ACCENT       = "#4FC3F7"
ACCENT2      = "#81C784"
WARN         = "#FFB74D"

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=LABEL_COLOR, labelsize=8)
    ax.spines[:].set_color(GRID_COLOR)
    ax.title.set_color(TITLE_COLOR)
    ax.title.set_fontsize(10)
    ax.title.set_fontweight("bold")
    ax.set_title(title, pad=8)
    ax.yaxis.label.set_color(LABEL_COLOR)
    ax.xaxis.label.set_color(LABEL_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


# ── Figure title ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.955, "Behavioral Matching Engine — MVP Validation",
         ha="center", va="center", fontsize=16, fontweight="bold", color=TITLE_COLOR)
fig.text(0.5, 0.932, "Focus User: Fitness Archetype  |  Population: 50 synthetic users (10 per archetype)  |  All pillars validated",
         ha="center", va="center", fontsize=9, color=LABEL_COLOR)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Pillar 1: Lifestyle Vector Heatmap (focus user)
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "P1 · Lifestyle Vector — Focus User (Fitness)")

V_idf = idf_weight(focus["V"])
colors_bar = [arch_colors["fitness"] if v > 0.15 else ACCENT for v in V_idf]
bars = ax1.bar(CATEGORIES, V_idf, color=colors_bar, edgecolor="#0F1923", linewidth=0.5)
ax1.set_ylabel("IDF-Normalized Weight", fontsize=8)
ax1.set_xticklabels(CATEGORIES, rotation=35, ha="right", fontsize=7.5)

for bar, val in zip(bars, V_idf):
    if val > 0.02:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=TITLE_COLOR)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Pillar 1: Decay dynamics — one category over time
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "P1 · Vector Decay — Gym Signal Over Time")

# Simulate a user who stops going to gym at day 30
days = np.linspace(0, 90, 500)
gym_lambda = LAMBDA[0]  # 0.02
# Pre-stop: accumulate to ~1.0, post-stop: pure decay
gym_signal = np.where(days <= 30,
    1.0 * (1 - np.exp(-0.15 * days)),         # buildup phase
    (1.0 * (1 - np.exp(-0.15 * 30))) * np.exp(-gym_lambda * (days - 30))  # decay
)

ax2.plot(days, gym_signal, color=arch_colors["fitness"], linewidth=2)
ax2.axvline(30, color=WARN, linestyle="--", linewidth=1.2, alpha=0.8)
ax2.text(31, 0.85, "Last gym visit", color=WARN, fontsize=7.5)
ax2.fill_between(days, gym_signal, alpha=0.15, color=arch_colors["fitness"])
ax2.set_xlabel("Days", fontsize=8)
ax2.set_ylabel("Gym Signal Magnitude", fontsize=8)
ax2.set_xlim(0, 90)
ax2.set_ylim(0, 1.05)

# Half-life annotation
half_life = np.log(2) / gym_lambda
ax2.axhline(gym_signal[days <= 30].max() / 2, color=ACCENT, linestyle=":", linewidth=1, alpha=0.6)
ax2.text(65, gym_signal[days <= 30].max() / 2 + 0.02, f"Half-life ≈ {half_life:.0f}d", 
         color=ACCENT, fontsize=7.5)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Pillar 2: Behavioral vs Naive similarity distributions
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "P2 · Behavioral vs. Naive Similarity Distribution")

beh_scores  = [r["behavioral"] for r in results]
naive_scores = [r["naive"]     for r in results]

bins = np.linspace(0, 1, 18)
ax3.hist(naive_scores, bins=bins, alpha=0.55, color=WARN,    label="Naive (tag overlap)",  edgecolor="#0F1923")
ax3.hist(beh_scores,   bins=bins, alpha=0.70, color=ACCENT,  label="Behavioral (cosine)",  edgecolor="#0F1923")
ax3.set_xlabel("Similarity Score", fontsize=8)
ax3.set_ylabel("Count", fontsize=8)
ax3.legend(fontsize=7.5, facecolor=PANEL_BG, labelcolor=TITLE_COLOR, edgecolor=GRID_COLOR)

ax3.axvline(np.mean(beh_scores),  color=ACCENT, linewidth=1.5, linestyle="--")
ax3.axvline(np.mean(naive_scores), color=WARN,  linewidth=1.5, linestyle="--")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — Behavioral similarity by archetype (box plot)
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "P2 · Behavioral Similarity by Archetype vs. Fitness")

arch_data = {a: [] for a in ARCHETYPES if a != "fitness"}
for r in results:
    if r["archetype"] in arch_data:
        arch_data[r["archetype"]].append(r["behavioral"])

archs  = list(arch_data.keys())
data   = [arch_data[a] for a in archs]
colors = [arch_colors[a] for a in archs]

bp = ax4.boxplot(data, patch_artist=True, medianprops=dict(color="white", linewidth=2),
                 whiskerprops=dict(color=LABEL_COLOR), capprops=dict(color=LABEL_COLOR),
                 flierprops=dict(markerfacecolor=LABEL_COLOR, markersize=4))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax4.set_xticklabels([a.capitalize() for a in archs], fontsize=8)
ax4.set_ylabel("Behavioral Similarity", fontsize=8)
ax4.axhline(0.5, color=ACCENT, linewidth=1, linestyle=":", alpha=0.6)
ax4.text(4.55, 0.515, "0.5", color=ACCENT, fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Pillar 3: Adaptive weight schedule
# ══════════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "P3 · Adaptive Weight Schedule vs. n_eff")

n_range = np.arange(0, 121)
w1_vals, w2_vals, w3_vals = [], [], []
for n in n_range:
    w1, w2, w3 = adaptive_weights(n, n)  # symmetric case
    w1_vals.append(w1)
    w2_vals.append(w2)
    w3_vals.append(w3)

ax5.stackplot(n_range, w3_vals, w2_vals, w1_vals,
              labels=["w₃ Bio (prior)", "w₂ Proximity (fixed)", "w₁ Behavioral"],
              colors=[WARN, "#78909C", ACCENT2], alpha=0.85)

ax5.set_xlabel("n_eff = min(n_A, n_B)  [ping count]", fontsize=8)
ax5.set_ylabel("Weight", fontsize=8)
ax5.set_xlim(0, 120)
ax5.set_ylim(0, 1)
ax5.legend(fontsize=7.5, loc="center right", facecolor=PANEL_BG,
           labelcolor=TITLE_COLOR, edgecolor=GRID_COLOR)

# Crossover annotation
crossover = next(i for i, (w1, _, w3) in enumerate(zip(w1_vals, w2_vals, w3_vals)) if w1 >= w3)
ax5.axvline(crossover, color="white", linewidth=1, linestyle="--", alpha=0.5)
ax5.text(crossover + 1, 0.88, f"Beh > Bio\nat n={crossover}", color="white", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — Holistic score evolution (one matched pair, growing n_eff)
# ══════════════════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, "P3 · Holistic Score G — Same Pair, Growing n_eff")

# Best behavioral match and a deceptive naive match
best_beh_idx  = max(range(len(results)), key=lambda i: results[i]["behavioral"])
best_naive_idx = max(range(len(results)), key=lambda i: results[i]["naive"] - results[i]["behavioral"])

pairs = [
    (best_beh_idx,  "High Beh Match (Fitness↔Fitness)", arch_colors["fitness"]),
    (best_naive_idx, "High Naive Match (Bio-only aligned)", WARN),
]

n_axis = np.arange(0, 121, 5)
for idx, label, color in pairs:
    r = results[idx]
    other = users[idx + 1]
    scores = []
    for n_eff in n_axis:
        w1, w2, w3 = adaptive_weights(n_eff, n_eff)
        g = (w1 * r["behavioral"] + w2 * proximity_score(
             (focus["commute"] + other["commute"]) / 2) + w3 * r["naive"])
        scores.append(g)
    ax6.plot(n_axis, scores, color=color, linewidth=2, label=label)
    ax6.scatter([n_axis[-1]], [scores[-1]], color=color, s=40, zorder=5)

ax6.set_xlabel("n_eff [ping count]", fontsize=8)
ax6.set_ylabel("Holistic Score G", fontsize=8)
ax6.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TITLE_COLOR, edgecolor=GRID_COLOR)
ax6.set_xlim(0, 120)
ax6.set_ylim(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 7 — Venue Recommendation: top matches for best pair
# ══════════════════════════════════════════════════════════════════════════════
ax7 = fig.add_subplot(gs[2, 0:2])
style_ax(ax7, "P4 · Venue Recommendation Scores — Best Matched Pair")

best_user = users[best_beh_idx + 1]
venue_scores = []
for v in VENUES:
    score = venue_score(focus["V"], best_user["V"], v)
    venue_scores.append((v["name"], score, v["partner"], v["category"]))

venue_scores.sort(key=lambda x: (x[1] is not None, x[1] if x[1] else -1), reverse=True)

names   = [vs[0] for vs in venue_scores]
scores  = [vs[1] if vs[1] is not None else 0 for vs in venue_scores]
gated   = [vs[1] is None for vs in venue_scores]
partner = [vs[2] for vs in venue_scores]
cat_idx = [vs[3] for vs in venue_scores]

bar_colors = []
for g, p in zip(gated, partner):
    if g:
        bar_colors.append("#37474F")
    elif p:
        bar_colors.append(ACCENT2)
    else:
        bar_colors.append(ACCENT)

bars7 = ax7.barh(names, scores, color=bar_colors, edgecolor="#0F1923", height=0.6)
ax7.axvline(THETA, color=WARN, linestyle="--", linewidth=1.2, alpha=0.8)
ax7.text(THETA + 0.003, -0.6, f"θ = {THETA}", color=WARN, fontsize=7.5)
ax7.set_xlabel("Venue Score V(v)", fontsize=8)
ax7.set_xlim(0, 0.7)
ax7.invert_yaxis()

for bar, score, g, p in zip(bars7, scores, gated, partner):
    label = "GATED" if g else (f"{score:.3f}" + (" ★" if p else ""))
    color = "#607D8B" if g else TITLE_COLOR
    ax7.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             label, va="center", fontsize=8, color=color)

# Legend
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor=ACCENT,  label="Recommended (organic)"),
    Patch(facecolor=ACCENT2, label="Recommended (partner ★)"),
    Patch(facecolor="#37474F", label="Hard-gated (low intersection)"),
]
ax7.legend(handles=legend_els, fontsize=7.5, facecolor=PANEL_BG,
           labelcolor=TITLE_COLOR, edgecolor=GRID_COLOR, loc="lower right")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 8 — Naive vs Behavioral ranking comparison (rank scatter)
# ══════════════════════════════════════════════════════════════════════════════
ax8 = fig.add_subplot(gs[2, 2])
style_ax(ax8, "P2 · Rank Shift: Naive → Behavioral")

beh_ranked   = sorted(range(len(results)), key=lambda i: results[i]["behavioral"],  reverse=True)
naive_ranked = sorted(range(len(results)), key=lambda i: results[i]["naive"],        reverse=True)

beh_rank  = {idx: rank+1 for rank, idx in enumerate(beh_ranked)}
naive_rank = {idx: rank+1 for rank, idx in enumerate(naive_ranked)}

for idx, r in enumerate(results):
    br = beh_rank[idx]
    nr = naive_rank[idx]
    color = arch_colors[r["archetype"]]
    ax8.scatter(nr, br, color=color, alpha=0.75, s=45, edgecolors="none")

ax8.plot([1, len(results)], [1, len(results)], color="white", linewidth=0.8,
         linestyle="--", alpha=0.3)
ax8.set_xlabel("Naive Rank (tag overlap)", fontsize=8)
ax8.set_ylabel("Behavioral Rank (cosine)", fontsize=8)
ax8.set_xlim(0, len(results) + 1)
ax8.set_ylim(0, len(results) + 1)

legend_els8 = [Patch(facecolor=arch_colors[a], label=a.capitalize()) for a in ARCHETYPES]
ax8.legend(handles=legend_els8, fontsize=7, facecolor=PANEL_BG,
           labelcolor=TITLE_COLOR, edgecolor=GRID_COLOR, loc="lower right")
ax8.text(2, len(results) - 3, "Points above line:\nbetter behaviorally\nthan naively",
         fontsize=6.5, color=LABEL_COLOR)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_path = "matching_engine_mvp.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
plt.close()
