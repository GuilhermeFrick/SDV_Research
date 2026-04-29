/**
 * @file vde_service.cpp
 * @brief ECU VDE (Vehicle Dynamics) — SOME/IP Service Provider
 *
 * Service ID : 0x1236  Instance ID: 0x0001
 * Event ID   : 0x8003 (VDE_DATA)   EventGroup : 0x0003
 *
 * Payload (16 bytes):
 *   [0-3]   float speed          (km/h)
 *   [4-7]   float steering_angle (graus, positivo = direita)
 *   [8-11]  float throttle       (0.0 – 100.0 %)
 *   [12-15] float brake          (0.0 – 100.0 %)
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <set>
#include <thread>
#include <vsomeip/vsomeip.hpp>

static constexpr vsomeip::service_t    SERVICE_ID  = 0x1236;
static constexpr vsomeip::instance_t   INSTANCE_ID = 0x0001;
static constexpr vsomeip::event_t      EVENT_VDE   = 0x8003;
static constexpr vsomeip::eventgroup_t EVTGRP_VDE  = 0x0003;

static std::atomic<bool>                     running{true};
static std::shared_ptr<vsomeip::application> app;

void signal_handler(int) { running = false; app->stop(); }

static std::vector<vsomeip::byte_t> pack_vde(float speed, float steer,
                                              float throttle, float brake)
{
    std::vector<vsomeip::byte_t> buf(16);
    float vals[4] = {speed, steer, throttle, brake};
    std::memcpy(buf.data(), vals, 16);
    return buf;
}

int main()
{
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    app = vsomeip::runtime::get()->create_application("VDE_Service");
    if (!app->init()) {
        std::cerr << "[VDE] Falha ao inicializar vSomeIP\n";
        return 1;
    }

    std::set<vsomeip::eventgroup_t> groups{EVTGRP_VDE};
    app->offer_event(SERVICE_ID, INSTANCE_ID, EVENT_VDE, groups,
                     vsomeip::event_type_e::ET_FIELD);
    app->offer_service(SERVICE_ID, INSTANCE_ID);
    std::cout << "[VDE] Serviço SOME/IP 0x1236 disponível\n";

    std::thread publisher([&]() {
        double t = 0.0;
        while (running) {
            // Simula aceleração, curva e frenagem
            float speed    = 60.0f + 20.0f * static_cast<float>(std::sin(t * 0.1));
            float steer    = 15.0f * static_cast<float>(std::sin(t * 0.3));
            float throttle = 40.0f + 10.0f * static_cast<float>(std::cos(t * 0.2));
            float brake    = (throttle < 35.0f) ? 20.0f : 0.0f;
            t += 0.05;

            auto payload = vsomeip::runtime::get()->create_payload();
            payload->set_data(pack_vde(speed, steer, throttle, brake));
            app->notify(SERVICE_ID, INSTANCE_ID, EVENT_VDE, payload);

            std::cout << "[VDE] speed=" << speed << " km/h  steer=" << steer
                      << "°  throttle=" << throttle << "%  brake=" << brake << "%\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    app->start();
    publisher.join();
    return 0;
}
