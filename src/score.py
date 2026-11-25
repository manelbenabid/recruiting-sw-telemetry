import numpy as np
import pandas as pd
from src.segment import find_corners, build_zones
import plotly.graph_objects as go


                                

def score_function_s1(output: pd.DataFrame, labels: pd.DataFrame):
    """the function takes the estimates segmentation of a tracks for metre: s, type(0=straght, 1=corner) and the true
    labels of that track and returns a score for how good that segmentation is. the higher the score the better the segmentation.

    Args:
        pd (dataframw): output: is the output of the segmentation function.
        pd (dataframe): labels: is the true segmentation labels for that tracks.
    """
    out = output.copy()
    true_labels = labels.copy()

    # sort the s column
    out = out.sort_values(by=["s"])
    true_labels = true_labels.sort_values(by=["s"])

    # the score formula: score [0-1] of how many metres were lables correctly
    # score = (1/track_length_m) * (sum(y_hat = y))
    d = len(labels["s"])
    y = out["type"]
    y_hat = true_labels["type"]

    score = (1/d) * (np.sum(y == y_hat))

    return score


def score_function_s2(output: pd.DataFrame, labels: pd.DataFrame):
    """the function takes the estimates segmentation blabla ill edit it later
    """
    out = output.copy()
    true_labels = labels.copy()

    # sort the s column
    out = out.sort_values(by=["s"]).reset_index(drop=True)
    true_labels = true_labels.sort_values(by=["s"]).reset_index(drop=True)

    # the score formula: score [0-1] of how many metres were lables correctly
    # score = (1/track_length_m) * (sum(type_hat = type AND zone_hat = zone))
    d = len(labels["s"])
    type_ = out["type"]
    type_hat = true_labels["type"]
    zone_ = out["zone"]
    zone_hat = true_labels["zone"]

    score = np.sum((type_ == type_hat) & (zone_ == zone_hat)) / d

    return score

def pair_zones(zone_hat: np.ndarray, type_hat: np.ndarray,
               zone: np.ndarray, type_: np.ndarray) -> dict[int, int]:
    """
    Pair each output zone with one label zone, largest overlap first.
    Only zones of the same type can be paired, and each zone is used once.
    Returns {output zone: label zone}.
    """
    same_type = type_hat == type_
    overlap = (pd.DataFrame({"zone_hat": zone_hat[same_type], "zone": zone[same_type]})
               .value_counts()                      # metres shared by each (output, label) pair
               .reset_index(name="metres")
               .sort_values("metres", ascending=False))

    pairs = {}
    used = set()
    for z_hat, z, _ in overlap.itertuples(index=False):
        if z_hat not in pairs and z not in used:
            pairs[z_hat] = z
            used.add(z)
    return pairs


def score_function_s3(output: pd.DataFrame, labels: pd.DataFrame) -> float:
    """
    Share of metres (0 to 1) where the output has the right type
    and its zone is paired with the label zone.
    output and labels: columns s, type (straight = 0, corner = 1), zone.
    """
    out = output.sort_values("s").reset_index(drop=True)
    lab = labels.sort_values("s").reset_index(drop=True)
    if not np.array_equal(out["s"].to_numpy(), lab["s"].to_numpy()):
        raise ValueError("output and labels are not on the same s grid")

    # hat = estimate (the segmenter's output), no hat = labels
    type_hat = out["type"].to_numpy()
    zone_hat = out["zone"].to_numpy()
    type_ = lab["type"].to_numpy()
    zone = lab["zone"].to_numpy()

    pairs = pair_zones(zone_hat, type_hat, zone, type_)
    zone_mapped = pd.Series(zone_hat).map(pairs).to_numpy()   # unpaired zone → NaN → counts as wrong

    correct = (type_hat == type_) & (zone_mapped == zone)
    return correct.sum() / len(out)

def compute_scores_s1(n: int, space: pd.DataFrame, cl: pd.DataFrame, labels):
    sp = space.copy()
    scores = []
    for i in range(n):
        seg = find_corners(cl = cl, window=space["smooth_window_m"][i], k_straight=space["k_straight"][i], 
                            min_straight_m= space["min_straight_m"][i], k_corner= space["k_corner"][i], k_link=  space["k_link"][i], max_link_m = space["max_link_m"][i])
        out = seg[0][["s", "is_straight"]]
        # straight = 0, corner = 1
        out['type'] = np.where(out['is_straight'], 0, 1)
        out = out[["s", "type"]]
        score = score_function_s1(output=out, labels=labels)
        scores.append(score)
    sp["score"] = scores
    sp = sp.sort_values(by=["score"], ascending=False, ignore_index=True)

    return sp.head(20)



def compute_scores_s2( n: int, space: pd.DataFrame, cl: pd.DataFrame, labels):
    sp = space.copy()
    scores = []
    for i in range(n):
        p = space.iloc[i]
        cl_, apexes = find_corners(cl=cl, window=int(p["smooth_window_m"]), k_straight=p["k_straight"],
                                    min_straight_m=p["min_straight_m"], k_corner=p["k_corner"],
                                   k_link=p["k_link"], max_link_m=p["max_link_m"])
        cl_, zones = build_zones(cl_, apexes)
        out = cl_[["s", "is_straight", "zone"]].copy()
        out['type'] = np.where(out['is_straight'], 0, 1)
        out = out[["s", "type", "zone"]]
        scores.append(score_function_s2(output=out, labels=labels))
    sp["score"] = scores
    sp = sp.sort_values(by=["score"], ascending=False, ignore_index=True)

    return sp.head(20)

def compute_scores_s3(n: int, space: pd.DataFrame, cl: pd.DataFrame, labels):
    sp = space.copy()
    scores = []
    for i in range(n):
        p = space.iloc[i]
        cl_, apexes = find_corners(cl=cl, window=int(p["smooth_window_m"]), k_straight=p["k_straight"],
                                    min_straight_m=p["min_straight_m"], k_corner=p["k_corner"],
                                   k_link=p["k_link"], max_link_m=p["max_link_m"])
        cl_, zones = build_zones(cl_, apexes)
        out = cl_[["s", "zone"]].copy()
        out["type"] = np.where(cl_["is_straight"], 0, 1)   # straight = 0, corner = 1
        scores.append(score_function_s3(output=out, labels=labels))
    sp["score"] = scores
    sp = sp.sort_values(by=["score"], ascending=False, ignore_index=True)

    return sp.head(20)

def plot_best_score(space_scores: pd.DataFrame, cl: pd.DataFrame):
    i = 0
    print("space:\n", space_scores.iloc[i])
    cl_seg, apexes = find_corners(cl = cl, window=space_scores["smooth_window_m"][i], k_straight=space_scores["k_straight"][i], 
                        min_straight_m= space_scores["min_straight_m"][i], k_corner= space_scores["k_corner"][i], k_link=  space_scores["k_link"][i], max_link_m = space_scores["max_link_m"][i])

    cl_seg["type"] = np.where(cl_seg["is_straight"], "straight", "corner")
    colors = {"straight": "blue", "corner": "red"}

    fig = go.Figure()
    for t, c in colors.items():
        part = cl_seg[cl_seg["type"] == t]
        fig.add_trace(go.Scatter(
            x=part["x"], y=part["y"],
            mode="markers",
            marker=dict(color=c, size=4),
            name=t,
            customdata=np.stack([part["s"], part["curvature_smooth"]], axis=-1),
            hovertemplate="s = %{customdata[0]:.0f} m<br>smoothed curvature = %{customdata[1]:.4f} 1/m<extra></extra>",
        ))

    fig.update_layout(
        title=f"Straights (|curvature| < {round(space_scores["k_straight"][i], ndigits=5)} for ≥ {space_scores["min_straight_m"][i]} m) vs corners",
        xaxis_title="x (m, east)",
        yaxis_title="y (m, north)",
        legend=dict(orientation="h", y=-0.12),
        width=1000, height=700,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # same scale on both axes so the track isn't stretched
    fig.show()
