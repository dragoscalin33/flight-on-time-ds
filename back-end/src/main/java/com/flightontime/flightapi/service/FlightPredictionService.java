package com.flightontime.flightapi.service;

import com.flightontime.flightapi.domain.flight.dto.FlightPredictionResponse;
import com.flightontime.flightapi.domain.flight.dto.FlightRequest;
import com.flightontime.flightapi.infra.client.datascience.FlightPredictionMapper;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiRequest;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class FlightPredictionService {

    @Autowired
    private DataScienceClientService dataScienceClientService;

    @Autowired
    private AirportService airportService;

    @Autowired
    private AirlineService airlineService;

    public FlightPredictionResponse predictDelay(FlightRequest flightRequest) {
        DataScienceApiRequest apiRequest = buildDataScienceApiRequest(flightRequest);
        DataScienceApiResponse apiResponse = dataScienceClientService.callDataScienceApi(apiRequest);
        return FlightPredictionMapper.toFlightPredictionResponse(apiResponse);
    }

    private DataScienceApiRequest buildDataScienceApiRequest(FlightRequest flightRequest) {
        String airlineName = airlineService.getAirlineByName(flightRequest.airline()).name();
        String airportOriginFullName = airportService.getAirportByIataCode(flightRequest.origin()).fullName();
        String airportDestinationFullName = airportService.getAirportByIataCode(flightRequest.destination()).fullName();
        return FlightPredictionMapper.toDsApiRequest(
                flightRequest,
                airlineName,
                airportOriginFullName,
                airportDestinationFullName);
    }

}
