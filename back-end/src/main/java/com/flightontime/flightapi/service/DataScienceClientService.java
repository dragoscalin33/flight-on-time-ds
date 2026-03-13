package com.flightontime.flightapi.service;

import com.flightontime.flightapi.domain.exception.DataScienceApiOfflineException;
import com.flightontime.flightapi.infra.client.datascience.DataScienceClientInterface;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiRequest;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiResponse;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class DataScienceClientService {

    @Autowired
    private DataScienceClientInterface client;

    @CircuitBreaker(name = "externalService", fallbackMethod = "fallbackCallDataScienceApi")
    public DataScienceApiResponse callDataScienceApi(DataScienceApiRequest apiRequest) {
        return client.getFlightPrediction(apiRequest);
    }

    public DataScienceApiResponse fallbackCallDataScienceApi(DataScienceApiRequest apiRequest, Throwable t) {
        if(t instanceof CallNotPermittedException) {
            throw new DataScienceApiOfflineException("O circuito está aberto. Chamada ao serviço de Data Science interrompida");
        }

        throw new DataScienceApiOfflineException("O serviço de Data Science está fora do ar");
    }
}
