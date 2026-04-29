//  ECU IMU — SOME/IP Service Provider
//  Publica dados de acelerômetro e giroscópio a cada 50ms.
//
//  Service ID : 0x1235
//  Instance ID: 0x0001
//  Event ID   : 0x8002 (IMU_DATA)
//  EventGroup : 0x0002
//
//  Payload (24 bytes):
//    [0-3]   float accel_x (m/s²)
//    [4-7]   float accel_y
//    [8-11]  float accel_z
//    [12-15] float gyro_x  (rad/s)
//    [16-19] float gyro_y
//    [20-23] float gyro_z

#include <vsomeip/vsomeip.hpp>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#include <iostream>
#include <csignal>
#include <random>

static constexpr vsomeip::service_t    SERVICE_ID  = 0x1235;
static constexpr vsomeip::instance_t   INSTANCE_ID = 0x0001;
static constexpr vsomeip::event_t      EVENT_IMU   = 0x8002;
static constexpr vsomeip::eventgroup_t EVTGRP_IMU  = 0x0002;

static std::atomic<bool> running{true};
static std::shared_ptr<vsomeip::application> app;

void signal_handler(int) { running = false; app->stop(); }

static std::vector<vsomeip::byte_t> pack_imu(
    float ax, float ay, float az,
    float gx, float gy, float gz)
{
    std::vector<vsomeip::byte_t> buf(24);
    float vals[6] = {ax, ay, az, gx, gy, gz};
    std::memcpy(buf.data(), vals, 24);
    return buf;
}

int main()
{
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.01f);

    app = vsomeip::runtime::get()->create_application("IMU_Service");
    if (!app->init()) {
        std::cerr << "[IMU] Falha ao inicializar vSomeIP\n";
        return 1;
    }

    std::set<vsomeip::eventgroup_t> groups{EVTGRP_IMU};
    app->offer_event(SERVICE_ID, INSTANCE_ID, EVENT_IMU, groups,
                     vsomeip::event_type_e::ET_FIELD);

    app->offer_service(SERVICE_ID, INSTANCE_ID);
    std::cout << "[IMU] Serviço SOME/IP 0x1235 disponível\n";

    std::thread publisher([&]() {
        double t = 0.0;
        while (running) {
            // Simula vibração + gravidade
            float ax = noise(rng);
            float ay = noise(rng);
            float az = 9.81f + noise(rng);
            float gx = 0.001f * static_cast<float>(std::sin(t)) + noise(rng);
            float gy = 0.001f * static_cast<float>(std::cos(t)) + noise(rng);
            float gz = noise(rng);
            t += 0.05;

            auto payload = vsomeip::runtime::get()->create_payload();
            payload->set_data(pack_imu(ax, ay, az, gx, gy, gz));
            app->notify(SERVICE_ID, INSTANCE_ID, EVENT_IMU, payload);

            std::cout << "[IMU] accel=(" << ax << "," << ay << "," << az
                      << ")  gyro=(" << gx << "," << gy << "," << gz << ")\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    app->start();
    publisher.join();
    return 0;
}
