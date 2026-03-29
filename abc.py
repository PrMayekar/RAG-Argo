import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


print("\n===================================")
print("   ARGO FLOAT ANALYTICS SYSTEM")
print("===================================\n")


# -------------------------------------------------
# LOAD NETCDF FILE
# -------------------------------------------------

folder = r"C:\Users\Ananya\Desktop\65\NetCDF-FIles\argo_files"

for f in os.listdir(folder):
    if f.endswith(".nc"):
        file_path = os.path.join(folder, f)
        break

print("Opening NetCDF File:", file_path)

data = nc.Dataset(file_path)

lat = np.array(data.variables["LATITUDE"][:])
lon = np.array(data.variables["LONGITUDE"][:])
temp = np.array(data.variables["TEMP"][:])
sal = np.array(data.variables["PSAL"][:])

print("\nTotal Float Positions:", len(lat))


# -------------------------------------------------
# FLOAT LOCATION MAP
# -------------------------------------------------

plt.figure()

plt.scatter(lon, lat)

plt.title("ARGO Float Locations")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid()

plt.show()


# -------------------------------------------------
# TRAJECTORY ANALYSIS
# -------------------------------------------------

print("\n========== TRAJECTORY ANALYSIS ==========")

time_steps = np.arange(len(lat)).reshape(-1,1)

model_lat = LinearRegression()
model_lon = LinearRegression()

model_lat.fit(time_steps, lat)
model_lon.fit(time_steps, lon)

future_steps = np.arange(len(lat), len(lat)+10).reshape(-1,1)

future_lat = model_lat.predict(future_steps)
future_lon = model_lon.predict(future_steps)


# -------------------------------------------------
# PRINT FUTURE PREDICTED COORDINATES
# -------------------------------------------------

print("\nFuture Predicted Coordinates:\n")

for i in range(len(future_lat)):
    print(f"Step {i+1}: Lat={future_lat[i]:.3f}, Lon={future_lon[i]:.3f}")


# -------------------------------------------------
# GRAPH 1 : PAST TRAJECTORY
# -------------------------------------------------

plt.figure()

plt.plot(lon,
         lat,
         marker="o",
         color="blue")

plt.title("Past ARGO Float Trajectory")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid()

plt.show()


# -------------------------------------------------
# GRAPH 2 : FUTURE TRAJECTORY ONLY
# -------------------------------------------------

plt.figure()

plt.plot(future_lon,
         future_lat,
         linestyle="--",
         marker="x",
         color="red")

plt.title("Predicted Future Float Trajectory")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid()

plt.show()


# -------------------------------------------------
# GRAPH 3 : PAST + FUTURE TRAJECTORY
# -------------------------------------------------

plt.figure()

plt.plot(lon,
         lat,
         marker="o",
         color="blue",
         label="Past Trajectory")

plt.plot(future_lon,
         future_lat,
         linestyle="--",
         marker="x",
         color="red",
         label="Predicted Future")

plt.plot([lon[-1], future_lon[0]],
         [lat[-1], future_lat[0]],
         linestyle=":",
         color="orange")

plt.title("Past vs Future Float Trajectory")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.legend()

plt.grid()

plt.show()


# -------------------------------------------------
# GRAPH 4 : 3D TRAJECTORY
# -------------------------------------------------

time_axis = np.arange(len(lat))
future_time = np.arange(len(lat), len(lat)+10)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot(lon,
        lat,
        time_axis,
        marker="o",
        label="Past Trajectory")

ax.plot(future_lon,
        future_lat,
        future_time,
        linestyle="--",
        marker="x",
        label="Predicted Future")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Time")

ax.set_title("3D Float Trajectory Prediction")

ax.legend()

plt.show()


# -------------------------------------------------
# TEMPERATURE PROFILE
# -------------------------------------------------

temp_profile = np.array(temp[0])
temp_profile = temp_profile[~np.isnan(temp_profile)]

depth = np.arange(len(temp_profile))

plt.figure()

plt.plot(temp_profile, depth)

plt.gca().invert_yaxis()

plt.title("Ocean Temperature Profile")

plt.xlabel("Temperature")
plt.ylabel("Depth")

plt.grid()

plt.show()


# -------------------------------------------------
# TEMPERATURE PREDICTION
# -------------------------------------------------

depth_scaled = (np.arange(len(temp_profile))/10).reshape(-1,1)

model = LinearRegression()

model.fit(depth_scaled, temp_profile)

future_depth = [[len(temp_profile)/10]]

prediction = model.predict(future_depth)

print("\nPredicted Temperature:", prediction)


# -------------------------------------------------
# FLOAT CLUSTERING
# -------------------------------------------------

coords = np.column_stack((lat, lon))

kmeans = KMeans(n_clusters=3)

labels = kmeans.fit_predict(coords)

plt.figure()

plt.scatter(lon, lat, c=labels)

plt.title("Float Formation Clusters")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid()

plt.show()


# -------------------------------------------------
# NEAREST FLOAT SEARCH
# -------------------------------------------------

nbrs = NearestNeighbors(n_neighbors=2)

nbrs.fit(coords)

distances, indices = nbrs.kneighbors(coords)

print("\nNearest Float Examples:\n")

print(indices[:5])


# -------------------------------------------------
# TEMPERATURE VS SALINITY
# -------------------------------------------------

sal_profile = np.array(sal[0])

valid_len = min(len(temp_profile), len(sal_profile))

plt.figure()

plt.scatter(temp_profile[:valid_len],
            sal_profile[:valid_len])

plt.title("Temperature vs Salinity")

plt.xlabel("Temperature")
plt.ylabel("Salinity")

plt.grid()

plt.show()


# -------------------------------------------------
# PCA VISUALIZATION
# -------------------------------------------------

features = np.column_stack((temp_profile[:50],
                            sal_profile[:50]))

pca = PCA(n_components=2)

reduced = pca.fit_transform(features)

plt.figure()

plt.scatter(reduced[:,0], reduced[:,1])

plt.title("PCA Ocean Feature Reduction")

plt.grid()

plt.show()


# -------------------------------------------------
# TEMPERATURE TREND
# -------------------------------------------------

series = pd.Series(temp_profile)

moving_avg = series.rolling(window=5).mean()

plt.figure()

plt.plot(temp_profile, label="Original Temperature")

plt.plot(moving_avg, label="Moving Average")

plt.title("Temperature Trend Analysis")

plt.legend()

plt.grid()

plt.show()


print("\n===================================")
print(" DEMO COMPLETED SUCCESSFULLY")
print("===================================")