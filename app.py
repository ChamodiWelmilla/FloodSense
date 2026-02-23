import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("flood_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

district_data = {
    "Ampara": {"lat": 7.2912, "lon": 81.6724, "elev": 12, "pop": 200, "urban": False, "landcover": "Agriculture", "soil": "Sandy"},
    "Anuradhapura": {"lat": 8.3114, "lon": 80.4037, "elev": 81, "pop": 150, "urban": False, "landcover": "Agriculture", "soil": "Loamy"},
    "Badulla": {"lat": 6.9934, "lon": 81.0550, "elev": 680, "pop": 250, "urban": False, "landcover": "Forest", "soil": "Loamy"},
    "Batticaloa": {"lat": 7.7102, "lon": 81.6924, "elev": 10, "pop": 180, "urban": False, "landcover": "Agriculture", "soil": "Sandy"},
    "Colombo": {"lat": 6.9271, "lon": 79.8612, "elev": 5, "pop": 3500, "urban": True, "landcover": "Urban", "soil": "Clay"},
    "Galle": {"lat": 6.0367, "lon": 80.2170, "elev": 15, "pop": 800, "urban": True, "landcover": "Urban", "soil": "Loamy"},
    "Gampaha": {"lat": 7.0840, "lon": 80.0098, "elev": 15, "pop": 1700, "urban": True, "landcover": "Urban", "soil": "Clay"},
    "Hambantota": {"lat": 6.1246, "lon": 81.1185, "elev": 16, "pop": 120, "urban": False, "landcover": "Scrub", "soil": "Sandy"},
    "Jaffna": {"lat": 9.6615, "lon": 80.0255, "elev": 5, "pop": 700, "urban": True, "landcover": "Urban", "soil": "Sandy"},
    "Kalutara": {"lat": 6.5854, "lon": 79.9607, "elev": 10, "pop": 1200, "urban": True, "landcover": "Urban", "soil": "Clay"},
    "Kandy": {"lat": 7.2906, "lon": 80.6337, "elev": 500, "pop": 1000, "urban": True, "landcover": "Urban", "soil": "Loamy"},
    "Kegalle": {"lat": 7.2513, "lon": 80.3464, "elev": 175, "pop": 300, "urban": False, "landcover": "Plantation", "soil": "Loamy"},
    "Kilinochchi": {"lat": 9.3803, "lon": 80.3983, "elev": 12, "pop": 100, "urban": False, "landcover": "Agriculture", "soil": "Sandy"},
    "Kurunegala": {"lat": 7.4863, "lon": 80.3623, "elev": 116, "pop": 350, "urban": False, "landcover": "Agriculture", "soil": "Loamy"},
    "Mannar": {"lat": 8.9810, "lon": 79.9044, "elev": 8, "pop": 80, "urban": False, "landcover": "Scrub", "soil": "Sandy"},
    "Matale": {"lat": 7.4675, "lon": 80.6234, "elev": 365, "pop": 250, "urban": False, "landcover": "Plantation", "soil": "Loamy"},
    "Matara": {"lat": 5.9549, "lon": 80.5550, "elev": 15, "pop": 600, "urban": True, "landcover": "Urban", "soil": "Loamy"},
    "Monaragala": {"lat": 6.8724, "lon": 81.3507, "elev": 150, "pop": 120, "urban": False, "landcover": "Agriculture", "soil": "Loamy"},
    "Mullaitivu": {"lat": 9.2671, "lon": 80.8142, "elev": 15, "pop": 90, "urban": False, "landcover": "Agriculture", "soil": "Sandy"},
    "Nuwara Eliya": {"lat": 6.9497, "lon": 80.7891, "elev": 1889, "pop": 400, "urban": False, "landcover": "Plantation", "soil": "Peaty"},
    "Polonnaruwa": {"lat": 7.9403, "lon": 81.0188, "elev": 60, "pop": 180, "urban": False, "landcover": "Agriculture", "soil": "Loamy"},
    "Puttalam": {"lat": 8.0330, "lon": 79.8260, "elev": 8, "pop": 150, "urban": False, "landcover": "Scrub", "soil": "Sandy"},
    "Ratnapura": {"lat": 6.6828, "lon": 80.3992, "elev": 130, "pop": 500, "urban": False, "landcover": "Plantation", "soil": "Loamy"},
    "Trincomalee": {"lat": 8.5711, "lon": 81.2335, "elev": 10, "pop": 200, "urban": False, "landcover": "Agriculture", "soil": "Sandy"},
    "Vavuniya": {"lat": 8.7542, "lon": 80.4982, "elev": 100, "pop": 120, "urban": False, "landcover": "Agriculture", "soil": "Loamy"},
}

def calculate_monthly_rainfall(weekly_rain):
    if weekly_rain == 0:
        return 0
    elif weekly_rain <= 200:
       
        return 120  
    else:
        # Extreme rainfall (>200mm): scale monthly to avoid contradiction
        # At 250mm weekly → 157.5mm monthly
        # At 300mm weekly → 195mm monthly
        # At 400mm weekly → 270mm monthly
        # At 500mm weekly → 345mm monthly
        # Formula: 120 + 75% of excess above 200mm
        excess = weekly_rain - 200
        monthly = 120 + (excess * 0.75)
        return min(monthly, 400) 

def get_saturation_adjusted_monthly(saturation_level):
    saturation_map = {
        "Dry": 150,      
        "Normal": 256,   
        "Saturated": 350, 
        "Extreme": 420  
    }
    return saturation_map.get(saturation_level, 256)

def adjust_flood_risk(base_prob, history, rain_7d, river_dist, elevation, built_up, pop_density):
    """
    Apply logical adjustments to ensure flood risk increases with risk factors.
    This corrects any counterintuitive model predictions.
    """
    adjusted_prob = base_prob

    if history > 0:
        if history <= 5:
            history_boost = history * 0.015 
        elif history <= 15:
            history_boost = 0.075 + (history - 5) * 0.02  
        else:
            history_boost = 0.275 + (history - 15) * 0.01  
            history_boost = min(history_boost, 0.45) 
        adjusted_prob += history_boost
    

    if rain_7d > 200: 
        rain_boost = min((rain_7d - 200) * 0.0005, 0.15)  
        adjusted_prob += rain_boost
    elif rain_7d > 100:  
        rain_boost = (rain_7d - 100) * 0.0002  
        adjusted_prob += rain_boost

    if river_dist < 500:
        proximity_boost = 0.10 - (river_dist / 500) * 0.05
        adjusted_prob += proximity_boost
    elif river_dist < 2000:
        proximity_boost = 0.05 * (1 - (river_dist - 500) / 1500)
        adjusted_prob += proximity_boost
    
    if elevation < 50: 
        elevation_boost = 0.08 * (1 - elevation / 50)  
        adjusted_prob += elevation_boost
    elif elevation < 150:  
        elevation_boost = 0.03 * (1 - (elevation - 50) / 100)  
        adjusted_prob += elevation_boost
    
    if built_up > 60 and pop_density > 1000:
        urban_boost = 0.05 
        adjusted_prob += urban_boost
    elif built_up > 40:
        urban_boost = 0.025  
        adjusted_prob += urban_boost
    
    extreme_factors = 0
    if history > 10:
        extreme_factors += 1
    if rain_7d > 250:
        extreme_factors += 1
    if river_dist < 1000:
        extreme_factors += 1
    if elevation < 30:
        extreme_factors += 1
    
    if extreme_factors >= 3:
        adjusted_prob += 0.05  

    max_increase = base_prob + 0.70
    adjusted_prob = min(adjusted_prob, max_increase)
    
    adjusted_prob = max(0.0, min(0.95, adjusted_prob))
   
    if history > 15 and rain_7d > 300 and river_dist < 1000:
        adjusted_prob = max(adjusted_prob, 0.60) 
    elif history > 10 and rain_7d > 200:
        adjusted_prob = max(adjusted_prob, 0.45)  
    elif history > 5 and rain_7d > 150:
        adjusted_prob = max(adjusted_prob, 0.32) 
    
    return adjusted_prob

st.set_page_config(page_title="FloodSense", page_icon="🌊", layout="wide")
st.title("🌊 FloodSense")

with st.sidebar:
    st.header("📍 Location & Weather")
    selected_dist = st.selectbox("Select District", sorted(district_data.keys()))
    
    geo = district_data[selected_dist]
    
    rain_7d = st.slider("7-Day Cumulative Rainfall (mm)", 0, 500, 50, 
                        help="Total rainfall in the past 7 days")
    
    river_dist = st.slider("Distance to Nearest River (m)", 0, 15000, 1000, 100,
                           help="Closer to river = higher flood risk")
    
    history = st.number_input("Historical Flood Events (past 10 years)", 0, 50, 0,
                              help="Number of times this area flooded in last 10 years")
    
    st.header("🏘️ Area Characteristics")
    
    use_defaults = st.checkbox("Use District Defaults", value=True,
                               help="Automatically use typical values for this district")
    
    if use_defaults:
        pop_density = geo["pop"]
        built_up = 80 if geo["urban"] else 15
        landcover = geo["landcover"]
        soil_type = geo["soil"]
        area_type = "Urban" if geo["urban"] else "Rural"
        
        st.info(f"**Auto-detected for {selected_dist}:**\n"
                f"- Population: {pop_density}/km²\n"
                f"- Built-up: {built_up}%\n"
                f"- Land: {landcover}\n"
                f"- Soil: {soil_type}\n"
                f"- Type: {area_type}")
    else:
        with st.expander("⚙️ Custom Area Settings", expanded=True):
            pop_density = st.number_input("Population Density (per km²)", 50, 4000, geo["pop"])
            built_up = st.slider("Built-up Area %", 0, 100, 80 if geo["urban"] else 15)
            
            landcover = st.selectbox(
                "Land Cover Type",
                ["Agriculture", "Forest", "Urban", "Wetland", "Plantation", "Scrub", "Bare soil"],
                index=["Agriculture", "Forest", "Urban", "Wetland", "Plantation", "Scrub", "Bare soil"].index(geo["landcover"])
            )
            
            soil_type = st.selectbox(
                "Soil Type",
                ["Loamy", "Sandy", "Clay", "Silty", "Peaty"],
                index=["Loamy", "Sandy", "Clay", "Silty", "Peaty"].index(geo["soil"])
            )
            
            area_type = st.radio("Settlement Type", ["Rural", "Urban"], 
                                index=1 if geo["urban"] else 0)

# Main Dashboard Layout
latitude = geo["lat"]
longitude = geo["lon"]
elevation = geo["elev"]

col_map, col_res = st.columns([1, 1])

with col_res:
    st.subheader("🌍 Geographic Details")
    
    use_default_location = st.checkbox("Use District Location", value=True,
                                       help="Use default coordinates and elevation for selected district")
    
    if use_default_location:
        latitude = geo["lat"]
        longitude = geo["lon"]
        elevation = geo["elev"]
        
        terrain = "Mountainous" if elevation > 500 else "Coastal/Low-lying" if elevation < 50 else "Plains"
        st.info(f"📍 {latitude:.4f}°N, {longitude:.4f}°E  \n⛰️ {elevation}m elevation  \n🏞️ {terrain}")
    else:
        latitude = st.number_input("Latitude (°N)", 5.0, 10.0, geo["lat"], 0.0001, format="%.4f",
                                  help="Decimal degrees North")
        longitude = st.number_input("Longitude (°E)", 79.0, 82.0, geo["lon"], 0.0001, format="%.4f",
                                   help="Decimal degrees East")
        elevation = st.number_input("Elevation (m)", 0, 2500, geo["elev"], 1,
                                   help="Meters above sea level")
    
    st.markdown("---")
    st.subheader("🎯 Risk Assessment")

    input_dict = dict.fromkeys(feature_columns, 0.0)
    monthly_rain = calculate_monthly_rainfall(rain_7d)

    input_dict.update({
        "latitude": latitude,
        "longitude": longitude,
        "elevation_m": elevation,
        "rainfall_7d_mm": rain_7d, 
        "monthly_rainfall_mm": monthly_rain,  
        "historical_flood_count": history,
        "distance_to_river_m": river_dist,
        "population_density_per_km2": pop_density,
        "built_up_percent": built_up,
        "nearest_hospital_km": 5,  
        "nearest_evac_km": 5,
    })

    for col in feature_columns:
        if col.startswith("district_"):
            input_dict[col] = 0.0
    if f"district_{selected_dist}" in feature_columns:
        input_dict[f"district_{selected_dist}"] = 1.0

    for col in feature_columns:
        if col.startswith("landcover_"):
            input_dict[col] = 0.0
    if f"landcover_{landcover}" in feature_columns:
        input_dict[f"landcover_{landcover}"] = 1.0

    for col in feature_columns:
        if col.startswith("soil_type_"):
            input_dict[col] = 0.0
    if f"soil_type_{soil_type}" in feature_columns:
        input_dict[f"soil_type_{soil_type}"] = 1.0

    for col in feature_columns:
        if col.startswith("urban_rural_"):
            input_dict[col] = 0.0
    if f"urban_rural_{area_type}" in feature_columns:
        input_dict[f"urban_rural_{area_type}"] = 1.0

    for col in feature_columns:
        if col.startswith("water_supply_"):
            input_dict[col] = 0.0
    if area_type == "Urban":
        if "water_supply_Municipal" in feature_columns:
            input_dict["water_supply_Municipal"] = 1.0
    else:
        if "water_supply_Well" in feature_columns:
            input_dict["water_supply_Well"] = 1.0

    for col in feature_columns:
        if col.startswith("electricity_"):
            input_dict[col] = 0.0
    if "electricity_Grid" in feature_columns:
        input_dict["electricity_Grid"] = 1.0

    for col in feature_columns:
        if col.startswith("road_quality_"):
            input_dict[col] = 0.0
    if area_type == "Urban":
        if "road_quality_Good (Paved)" in feature_columns:
            input_dict["road_quality_Good (Paved)"] = 1.0
    else:
        if "road_quality_Fair" in feature_columns:
            input_dict["road_quality_Fair"] = 1.0

    input_df = pd.DataFrame([input_dict])[feature_columns]
    scaled_input = scaler.transform(input_df)
    base_prob = model.predict_proba(scaled_input)[0][1]
    
    prob = adjust_flood_risk(
        base_prob=base_prob,
        history=history,
        rain_7d=rain_7d,
        river_dist=river_dist,
        elevation=elevation,
        built_up=built_up,
        pop_density=pop_density
    )
    
    risk_color = "🔴" if prob > 0.60 else "🟡" if prob > 0.40 else "🟢"
    st.metric(
        "Flood Risk Probability",
        f"{prob:.1%}",
        help="ML model prediction with risk factor adjustments"
    )

    if prob > 0.60:
        st.error("🚨 **CRITICAL HIGH RISK**")
    elif prob > 0.30:
        st.warning("⚠️ **MODERATE RISK**")
    else:
        st.success("✅ **LOW RISK**")
    
    # Risk Factor Breakdown
    with st.expander("📊 Risk Factor Analysis", expanded=False):
        st.write("**Key Risk Contributors:**")
        
        if history > 15:
            st.write(f"🔴 **Historical Floods:** {history} events - Extremely flood-prone area")
        elif history > 10:
            st.write(f"🟠 **Historical Floods:** {history} events - Very high frequency")
        elif history > 5:
            st.write(f"🟡 **Historical Floods:** {history} events - High frequency")
        elif history > 0:
            st.write(f"🟢 **Historical Floods:** {history} events - Moderate frequency")
        else:
            st.write("✅ **Historical Floods:** No recorded events")
        
        if rain_7d > 300:
            st.write(f"🔴 **7-Day Rainfall:** {rain_7d}mm - Extreme precipitation")
        elif rain_7d > 200:
            st.write(f"🟠 **7-Day Rainfall:** {rain_7d}mm - Very heavy rainfall")
        elif rain_7d > 100:
            st.write(f"🟡 **7-Day Rainfall:** {rain_7d}mm - Heavy rainfall")
        elif rain_7d > 50:
            st.write(f"🟢 **7-Day Rainfall:** {rain_7d}mm - Moderate rainfall")
        else:
            st.write(f"✅ **7-Day Rainfall:** {rain_7d}mm - Light rainfall")
        
        if river_dist < 500:
            st.write(f"🔴 **River Distance:** {river_dist}m - Very close, high flood risk")
        elif river_dist < 2000:
            st.write(f"🟡 **River Distance:** {river_dist}m - Close to river")
        elif river_dist < 5000:
            st.write(f"🟢 **River Distance:** {river_dist}m - Moderate distance")
        else:
            st.write(f"✅ **River Distance:** {river_dist}m - Safe distance")
        
        if elevation < 30:
            st.write(f"🔴 **Elevation:** {elevation}m - Very low-lying area")
        elif elevation < 50:
            st.write(f"🟡 **Elevation:** {elevation}m - Low coastal area")
        elif elevation < 150:
            st.write(f"🟢 **Elevation:** {elevation}m - Plains")
        else:
            st.write(f"✅ **Elevation:** {elevation}m - Higher ground")
        
        if built_up > 60 and pop_density > 1000:
            st.write(f"🟠 **Urban Density:** {built_up}% built-up, {pop_density}/km² - Dense urban area with drainage challenges")
        elif built_up > 40:
            st.write(f"🟡 **Urban Density:** {built_up}% built-up - Moderate urbanization")
        else:
            st.write(f"🟢 **Urban Density:** {built_up}% built-up - Low urbanization")
        
        st.write("---")
        st.caption(f"Base Model Prediction: {base_prob:.1%} | Adjusted Risk: {prob:.1%}")

    st.write("---")
    st.caption(
        "⚡For the best results, please ensure that all input fields are completed accurately."
    )

with col_map:
    st.subheader(f"📍 {selected_dist} District")
    map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_data, zoom=9)