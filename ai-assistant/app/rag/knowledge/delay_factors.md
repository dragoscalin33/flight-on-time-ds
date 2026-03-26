# Flight Delay Factors in Brazilian Aviation

## Weather Factors (Primary Cause ~40% of delays)

### Precipitation
- Rain above 10mm/hour significantly increases delay probability
- Light drizzle (<2mm/hour) has minimal impact on operations
- Thunderstorms (common October-March) can shut down operations entirely
- Impact varies by airport: short-runway airports (CGH, SDU) more affected

### Wind
- Crosswinds above 25 km/h begin affecting operations
- Tailwinds above 15 km/h can require runway changes, causing delays
- Gusts above 40 km/h may temporarily halt operations
- Coastal airports (SSA, REC, GIG) more frequently affected by wind

### Visibility
- Fog (visibility <800m) is the second most common weather delay cause
- Most common June-August in São Paulo, Curitiba, Belo Horizonte
- Early morning fog typically clears by 10am
- Urban pollution can compound fog effects (especially CGH)

### Temperature
- Extreme heat (>40°C) can restrict takeoff weight
- Cold fronts from the south bring turbulence and storms
- Temperature inversions trap fog and pollution

## Operational Factors (~30% of delays)

### Air Traffic Congestion
- Peak hours at major airports create ground delays
- Ground delay programs during weather events cascade
- São Paulo airports (CGH/GRU) most congested nationally
- Holiday periods amplify congestion effects

### Aircraft Turnaround
- Cascading delays from late arriving aircraft
- Hub airports more susceptible to cascade effects
- LATAM's hub model particularly vulnerable
- Late evening flights accumulate delays from the entire day

### Technical Issues
- Aircraft maintenance requirements
- Ground equipment problems
- Crew duty time limitations
- Fueling delays

## Temporal Factors (~20% of delays)

### Time of Day
- Early morning (6-8am): fog risk but less congestion
- Mid-morning (9-11am): generally best performance
- Afternoon (14-17pm): thunderstorm risk (wet season)
- Evening (18-21pm): accumulated delays from earlier disruptions

### Day of Week
- Monday and Friday: highest delay rates (business travel peaks)
- Tuesday and Wednesday: lowest delay rates
- Sunday evening: elevated delays from weekend return traffic
- Saturday morning: moderate (leisure travel)

### Month
- December: worst month (weather + holiday traffic)
- January: second worst (continued holiday + wet season peak)
- September: typically best month (end of dry season, low traffic)
- June-July: moderate (fog season + school holidays)

## Geographic Factors (~10% of delays)

### Route Distance
- Longer routes (>2000km) have more buffer for time recovery
- Short routes (<500km) more affected by ground delays proportionally
- Regional routes with turboprops more weather-sensitive
- Shuttle routes (CGH-SDU) have highest frequency but tight scheduling

### Airport Infrastructure
- Runway length directly correlates with weather resilience
- Single-runway airports create bottlenecks
- Modern airports (BSB, GRU T3) have better ground operations
- Older airports (CGH, SDU) constrained by urban surroundings

## How the FlightOnTime Model Uses These Factors

The CatBoost model considers 11 features:
1. **Airline** - captures carrier-specific operational patterns
2. **Origin airport** - captures departure airport characteristics
3. **Destination airport** - captures arrival airport characteristics
4. **Distance (km)** - Haversine distance between airports
5. **Hour** - time of day departure
6. **Day of week** - weekday patterns
7. **Month** - seasonal patterns
8. **Holiday flag** - Brazilian national holidays
9. **Precipitation (mm)** - live weather from OpenMeteo
10. **Wind speed (km/h)** - live weather from OpenMeteo
11. **Weather imputation flag** - indicates if weather data was estimated

The model uses a safety-first threshold of 0.35 (instead of 0.50) to prioritize recall (90.8%), meaning it catches most actual delays at the cost of some false alerts.
