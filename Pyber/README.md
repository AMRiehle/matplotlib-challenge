
# Pyber
GWU Data Analytics Bootcamp Homework 5

* Generally speaking, the higher the number of rides, the lower the average fare.
* Despite having lower average fares, rides in Urban areas still bring in the highest total revenue.
* The more urban an area, the more business Pyber is likely to do in that area.


```python
# Import dependencies

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in City data, removing duplicate data 
# Was asked to keep one instance of duplicate city, so kept city instance with higher driver count

city_csv = "raw_data/city_data.csv"
city_df = pd.read_csv(city_csv)
city_df = city_df.loc[(city_df['city'] != "Port James") | (city_df['driver_count'] == 15), :]

# Read in Rides data

rides_csv = "raw_data/ride_data.csv"
ride_df = pd.read_csv(rides_csv)

# Merge data sets

df = pd.merge(city_df, ride_df, on="city")
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>driver_count</th>
      <th>type</th>
      <th>date</th>
      <th>fare</th>
      <th>ride_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-19 04:27:52</td>
      <td>5.51</td>
      <td>6246006544795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-04-17 06:59:50</td>
      <td>5.54</td>
      <td>7466473222333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-05-04 15:06:07</td>
      <td>30.54</td>
      <td>2140501382736</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-01-25 20:44:56</td>
      <td>12.08</td>
      <td>1896987891309</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kelseyland</td>
      <td>63</td>
      <td>Urban</td>
      <td>2016-08-09 18:19:47</td>
      <td>17.91</td>
      <td>8784212854829</td>
    </tr>
  </tbody>
</table>

### Bubble Plot of Ride Sharing Data


```python
# Filter and aggregate urban data in preparation for charting

urban_df = df.loc[df['type'] == 'Urban', :]
urban_fares = urban_df.groupby('city')['fare'].mean()
urban_rides = urban_df.groupby('city')['ride_id'].count()
urban_drivers = city_df.loc[city_df['type'] == 'Urban', ['city','driver_count']].set_index('city')

# Filter and aggregate suburban data in preparation for charting

suburban_df = df.loc[df['type'] == 'Suburban', :]
suburban_fares = suburban_df.groupby('city')['fare'].mean()
suburban_rides = suburban_df.groupby('city')['ride_id'].count()
suburban_drivers = city_df.loc[city_df['type'] == 'Suburban', ['city','driver_count']].set_index('city')

# Filter and aggregate rural data in preparation for charting

rural_df = df.loc[df['type'] == 'Rural', :]
rural_fares = rural_df.groupby('city')['fare'].mean()
rural_rides = rural_df.groupby('city')['ride_id'].count()
rural_drivers = city_df.loc[city_df['type'] == 'Rural', ['city','driver_count']].set_index('city')

# Create scatterplot and add data

fig, ax = plt.subplots()
plt.scatter(urban_rides, urban_fares, marker="o", facecolors="lightskyblue", edgecolor="blue", s=urban_drivers*8, label="Urban")
plt.scatter(suburban_rides, suburban_fares, marker="o", facecolors="lightcoral", edgecolor="red", s=suburban_drivers*8, label="Suburban")
plt.scatter(rural_rides, rural_fares, marker="o", facecolors="gold", s=rural_drivers*8, label="Rural", edgecolor="orange")

# Format scatterplot, removing outlier datapoint from plot via y-limits

lgnd = ax.legend(title="City Types")
lgnd.legendHandles[0]._sizes = [100]
lgnd.legendHandles[1]._sizes = [100]
lgnd.legendHandles[2]._sizes = [100]
plt.xlim(0, 40)
plt.ylim(20, 40)
ax.patch.set_facecolor('lightgray')
ax.patch.set_alpha(0.6)
plt.grid(linewidth=2, color='white', alpha=0.6)
ax.set_axisbelow(True)
plt.title("Pyber Ride Sharing Data", fontweight='bold', fontsize=14)
plt.xlabel("Total Number of Rides (Per City)", fontsize=12)
plt.ylabel("Average Fare ($)", fontsize=12)
print("Note: Circle size corresponds with driver counts per city.")
```

    Note: Circle size corresponds with driver counts per city.


![png](Images/output_4_1.png)


### Total Fares by City Type


```python
# Create pie chart

colors = ['gold', 'lightcoral', 'lightskyblue']
explode = [0, 0.02, 0.1]
fares_df = df.groupby('type')['fare'].sum()
fares_chart = fares_df.plot(kind='pie', colors=colors, explode=explode, shadow=True, figsize=(4.5,4.5), autopct="%1.1f%%", startangle=-40)
plt.title('Percent of Fares by City Type', fontweight='bold', fontsize=14)
plt.ylabel("")
```

![png](Images/output_6_1.png)

### Total Rides by City Type


```python
# Create pie chart

trips_df = df.groupby('type')['ride_id'].count()
trips_chart = trips_df.plot(kind='pie', colors=colors, explode=explode, shadow=True, figsize=(4.5,4.5), autopct="%1.1f%%", startangle=-22)
trips_chart.set_ylabel('')
plt.title('Percent of Rides by City Type', fontweight='bold', fontsize=14)
```

![png](Images/output_8_1.png)

### Total Drivers by City Type


```python
# Create pie chart

drivers_df = city_df.groupby('type')['driver_count'].sum()
drivers_chart = drivers_df.plot(kind='pie', colors=colors, explode=explode, shadow=True, figsize=(4.5,4.5), autopct="%1.1f%%")
drivers_chart.set_ylabel('')
plt.title('Percent of Drivers by City Type', fontweight='bold', fontsize=14)
```

![png](Images/output_10_1.png)

