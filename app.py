# app.py
import os
import math
import json
import time
import polyline
import requests
import streamlit as st

try:
    from streamlit_folium import st_folium
    import folium
    HAS_MAP = True
except Exception:
    HAS_MAP = False

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    client = OpenAI()
    # Attempt to detect if key is present; if not, we'll skip LLM features gracefully
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Smart Route & Mode Recommender", page_icon="🗺️", layout="wide")

st.title("🗺️ Smart Route & Mode Recommender")
st.caption("Free-API demo using OSRM + OSM for routing, and Google Maps for one-click directions. Optional: ChatGPT summary if an `OPENAI_API_KEY` is set.")

with st.sidebar:
    st.header("Inputs")
    origin = st.text_input("Origin", placeholder="e.g., IIT Kanpur, Kanpur")
    destination = st.text_input("Destination", placeholder="e.g., Lucknow Junction, Lucknow")
    city_hint = st.text_input("City/Area (optional)", placeholder="e.g., Kanpur, Bengaluru, Delhi")
    use_llm = st.toggle("Generate natural-language summary (uses your OpenAI API key if set)", value=False)
    st.markdown("---")
    st.subheader("Assumptions (edit if you like)")
    avg_speeds = {
        "car": st.number_input("Car avg speed (km/h)", 1, 200, 35),
        "motorcycle": st.number_input("Motorcycle avg speed (km/h)", 1, 200, 28),
        "bus": st.number_input("Bus avg speed (km/h)", 1, 200, 22),
        "metro": st.number_input("Metro avg speed (km/h)", 1, 200, 32),
        "bicycle": st.number_input("Bicycle avg speed (km/h)", 1, 60, 14),
        "foot": st.number_input("Walking speed (km/h)", 1, 15, 4),
    }
    wait_buffer_min = st.number_input("Transit wait/transfer buffer (min)", 0, 60, 8)

st.markdown("#### What this app does")
st.write(
    "- Finds a route between two places using **OSRM** (car 🚗, bicycle 🚲, walk 🚶). "
    "For bus/metro, it estimates travel time using average speeds + buffer.\n"
    "- Estimates **energy use (kWh per passenger-km)** for each mode and **energy saved** vs a single-occupancy car.\n"
    "- Opens **Google Maps** in one click for detailed, real-time directions (no API key needed for the web link)."
)

def geocode(query):
    # Nominatim geocoding (OpenStreetMap) - free use, rate-limited
    if not query:
        return None
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "streamlit-route-demo/1.0 (contact: example@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    item = data[0]
    return {
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
        "display_name": item.get("display_name", query),
    }

def osrm_route(profile, o, d):
    # profiles: car->driving, bicycle->cycling, foot->walking
    profile_map = {"car": "driving", "bicycle": "cycling", "foot": "walking"}
    profile_name = profile_map.get(profile, "driving")
    base = f"https://router.project-osrm.org/route/v1/{profile_name}/{o['lon']},{o['lat']};{d['lon']},{d['lat']}"
    params = {"overview": "full", "geometries": "polyline", "alternatives": "false"}
    r = requests.get(base, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data or "routes" not in data or not data["routes"]:
        return None
    route = data["routes"][0]
    return {
        "distance_km": route["distance"] / 1000.0,
        "duration_min": route["duration"] / 60.0,
        "geometry": route.get("geometry"),
    }

ENERGY_KWH_PER_KM = {
    # per passenger-km (rough heuristics for demo purposes)
    "car": 0.65,         # ICE, single occupancy
    "motorcycle": 0.20,  # petrol bike
    "bus": 0.15,         # per passenger-km (assuming typical occupancy)
    "metro": 0.06,       # per passenger-km
    "bicycle": 0.005,    # human metabolic equiv.
    "foot": 0.0,         # ignore metabolic for simplicity
}

def estimate_transit_time(distance_km, mode, avg_speeds, buffer_min):
    v = avg_speeds.get(mode, 20)
    base = (distance_km / max(v, 1e-6)) * 60.0  # minutes
    return base + buffer_min

def recommend_modes(distances):
    # Returns a dict with recommended fastest and greenest modes
    if not distances:
        return {}
    # Fastest
    fastest = min(distances.items(), key=lambda kv: kv[1]["duration_min"])[0]
    # Greenest (lowest energy)
    greenest = min(distances.items(), key=lambda kv: kv[1]["energy_kwh"])[0]
    return {"fastest": fastest, "greenest": greenest}

def google_maps_link(o, d, travelmode="driving"):
    # Opens Google Maps website with directions (no API key required for user navigation)
    def enc(s):
        return s.replace(" ", "+")
    return f"https://www.google.com/maps/dir/?api=1&origin={enc(o)}&destination={enc(d)}&travelmode={travelmode}"

if st.button("🔎 Calculate Route & Recommendations", type="primary"):
    query_o = origin if not city_hint else f"{origin}, {city_hint}"
    query_d = destination if not city_hint else f"{destination}, {city_hint}"
    with st.spinner("Geocoding places..."):
        o = geocode(query_o)
        d = geocode(query_d)
    if not o or not d:
        st.error("Couldn't geocode one or both locations. Please refine the inputs (add city/state/country).")
        st.stop()

    # OSRM for car/bike/foot
    distances = {}
    with st.spinner("Fetching routes (OSRM)..."):
        for mode in ["car", "bicycle", "foot"]:
            route = osrm_route(mode, o, d)
            if route:
                distances[mode] = {
                    "distance_km": route["distance_km"],
                    "duration_min": route["duration_min"],
                    "geometry": route.get("geometry"),
                }

    # If OSRM failed for all, bail
    if not distances:
        st.error("OSRM could not produce a route. Try different inputs.")
        st.stop()

    # For transit-like modes (bus/metro/motorcycle), estimate using distances from car/bicycle
    # Use car distance as base travel distance for motorized modes; bicycle for bicycle (already there)
    base_distance_km = distances.get("car", list(distances.values())[0])["distance_km"]
    # Motorcycle time ~ like motorcycle avg speed
    moto_time = estimate_transit_time(base_distance_km, "motorcycle", avg_speeds, buffer_min=0)
    distances["motorcycle"] = {"distance_km": base_distance_km, "duration_min": moto_time, "geometry": None}
    # Bus & Metro estimates
    bus_time = estimate_transit_time(base_distance_km, "bus", avg_speeds, wait_buffer_min)
    metro_time = estimate_transit_time(base_distance_km, "metro", avg_speeds, wait_buffer_min)
    distances["bus"] = {"distance_km": base_distance_km, "duration_min": bus_time, "geometry": None}
    distances["metro"] = {"distance_km": base_distance_km, "duration_min": metro_time, "geometry": None}

    # Compute energy per mode
    for mode, vals in distances.items():
        ek = ENERGY_KWH_PER_KM.get(mode, 0.2)
        vals["energy_kwh"] = ek * vals["distance_km"]

    # Energy saved vs car
    car_energy = distances["car"]["energy_kwh"] if "car" in distances else None
    for mode, vals in distances.items():
        if car_energy is not None:
            vals["energy_saved_vs_car_kwh"] = car_energy - vals["energy_kwh"]
        else:
            vals["energy_saved_vs_car_kwh"] = None

    # Recommendations
    recs = recommend_modes(distances)

    # Display metrics
    left, right = st.columns([1,1])
    with left:
        st.subheader("Recommendations")
        if recs:
            st.success(f"**Fastest:** {recs['fastest'].title()}")
            st.success(f"**Greenest:** {recs['greenest'].title()}")
        else:
            st.info("Could not compute recommendations.")

        if use_llm and OPENAI_AVAILABLE:
            try:
                summary_prompt = f"""Summarize route findings conversationally.
Origin: {origin}
Destination: {destination}
Distances/times/energy: {json.dumps(distances, indent=2)}
Recommendations: {json.dumps(recs, indent=2)}
Keep it under 120 words. Mention fastest and greenest explicitly.
"""
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user", "content": summary_prompt}],
                    temperature=0.3,
                )
                txt = resp.choices[0].message.content
                st.write("**AI Summary (ChatGPT):**")
                st.write(txt)
            except Exception as e:
                st.warning("LLM summary skipped (missing/invalid API key or network restricted).")
        elif use_llm and not OPENAI_AVAILABLE:
            st.warning("Set your OPENAI_API_KEY as a Streamlit secret or environment variable to enable the summary.")

    with right:
        st.subheader("Quick open in Google Maps")
        st.write("Click to open native, real-time directions:")
        gmodes = {
            "Driving (Google Maps)": "driving",
            "Transit (Google Maps)": "transit",
            "Walking (Google Maps)": "walking",
            "Bicycling (Google Maps)": "bicycling",
        }
        for label, tm in gmodes.items():
            st.link_button(label, google_maps_link(origin, destination, tm))

    st.markdown("---")
    st.subheader("Mode comparison")
    # Make a nice table
    import pandas as pd
    rows = []
    for mode, vals in distances.items():
        rows.append({
            "Mode": mode.title(),
            "Distance (km)": round(vals["distance_km"], 2),
            "Time (min)": round(vals["duration_min"], 1),
            "Energy (kWh)": round(vals["energy_kwh"], 2),
            "Energy Saved vs Car (kWh)": None if vals["energy_saved_vs_car_kwh"] is None else round(vals["energy_saved_vs_car_kwh"], 2),
        })
    df = pd.DataFrame(rows).sort_values(by="Time (min)")
    st.dataframe(df, hide_index=True)

    # Map
    st.markdown("---")
    st.subheader("Map (OSRM route geometry)")
    if HAS_MAP:
        # Choose the best available geometry (prefer car -> bicycle -> foot)
        geom = None
        chosen = None
        for k in ["car", "bicycle", "foot"]:
            if k in distances and distances[k].get("geometry"):
                geom = distances[k]["geometry"]
                chosen = k
                break
        if geom:
            coords = polyline.decode(geom)  # (lat, lon) pairs
            mid = coords[len(coords)//2]
            m = folium.Map(location=mid, zoom_start=11, control_scale=True)
            folium.Marker([o["lat"], o["lon"]], tooltip="Origin").add_to(m)
            folium.Marker([d["lat"], d["lon"]], tooltip="Destination").add_to(m)
            folium.PolyLine(coords, weight=5, opacity=0.8).add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, height=480, use_container_width=True)
            st.caption(f"Showing {chosen} route geometry from OSRM.")
        else:
            st.info("No polyline geometry available from OSRM for mapping.")
    else:
        st.info("Map dependencies not available. Install `folium` and `streamlit-folium` to see the route map.")

st.markdown("---")
st.caption("Notes: Energy figures are illustrative per-passenger-km averages for demo purposes and can vary widely by vehicle, traffic, and occupancy. OSRM public server has rate limits; for production, host your own or use a managed service.")