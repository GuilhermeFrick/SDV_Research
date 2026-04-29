//  ECU GPS — SOME/IP Service Provider
//  Publica coordenadas GPS como SOME/IP notifications a cada 100ms.
//
//  Service ID : 0x1234
//  Instance ID: 0x0001
//  Event ID   : 0x8001 (GPS_POSITION)
//  EventGroup : 0x0001
//
//  Payload (8 bytes):
//    [0-3] float latitude   (IEEE 754)
//    [4-7] float longitude  (IEEE 754)

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <thread>
#include <vsomeip/vsomeip.hpp>

static constexpr vsomeip::service_t    SERVICE_ID  = 0x1234;
static constexpr vsomeip::instance_t   INSTANCE_ID = 0x0001;
static constexpr vsomeip::event_t      EVENT_GPS   = 0x8001;
static constexpr vsomeip::eventgroup_t EVTGRP_GPS  = 0x0001;

static std::atomic<bool>                     running{true};
static std::shared_ptr<vsomeip::application> app;

void signal_handler(int)
{
    running = false;
    app->stop();
}

// Empacota dois floats em vetor de bytes (big-endian)
static std::vector<vsomeip::byte_t> pack_gps(float lat, float lon)
{
    std::vector<vsomeip::byte_t> buf(8);
    std::memcpy(buf.data(), &lat, 4);
    std::memcpy(buf.data() + 4, &lon, 4);
    return buf;
}

int main()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    app = vsomeip::runtime::get()->create_application("GPS_Service");
    if (!app->init())
    {
        std::cerr << "[GPS] Falha ao inicializar vSomeIP\n";
        return 1;
    }

    // Registra o evento GPS
    std::set<vsomeip::eventgroup_t> groups{EVTGRP_GPS};
    app->offer_event(SERVICE_ID, INSTANCE_ID, EVENT_GPS, groups, vsomeip::event_type_e::ET_FIELD);

    app->offer_service(SERVICE_ID, INSTANCE_ID);
    std::cout << "[GPS] Serviço SOME/IP 0x1234 disponível\n";

    // Thread de publicação
    std::thread publisher([&]() {
        double t = 0.0;
        while (running)
        {
            // Simula trajetória circular (latitude/longitude em graus)
            float lat = -23.5505f + 0.001f * static_cast<float>(std::sin(t));
            float lon = -46.6333f + 0.001f * static_cast<float>(std::cos(t));
            t += 0.01;

            auto payload = vsomeip::runtime::get()->create_payload();
            auto data    = pack_gps(lat, lon);
            payload->set_data(data);
            app->notify(SERVICE_ID, INSTANCE_ID, EVENT_GPS, payload);

            std::cout << "[GPS] lat=" << lat << "  lon=" << lon << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    app->start(); // bloqueia até app->stop()
    publisher.join();
    return 0;
}
