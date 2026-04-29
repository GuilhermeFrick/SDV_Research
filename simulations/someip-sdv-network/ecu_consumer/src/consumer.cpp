/**
 * @file consumer.cpp
 * @author  (frickoliveira.ee@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-04-29
 *
 * @copyright Copyright (c) 2026
 *
 */
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <vsomeip/vsomeip.hpp>

static constexpr vsomeip::service_t GPS_SERVICE = 0x1234;
static constexpr vsomeip::service_t IMU_SERVICE = 0x1235;
static constexpr vsomeip::instance_t INSTANCE_ID = 0x0001;
static constexpr vsomeip::event_t EVENT_GPS = 0x8001;
static constexpr vsomeip::event_t EVENT_IMU = 0x8002;
static constexpr vsomeip::eventgroup_t EVTGRP_GPS = 0x0001;
static constexpr vsomeip::eventgroup_t EVTGRP_IMU = 0x0002;

static std::atomic<bool> running{true};
static std::shared_ptr<vsomeip::application> app;

void signal_handler(int) {
  running = false;
  app->stop();
}

void on_message(const std::shared_ptr<vsomeip::message> &msg) {
  auto payload = msg->get_payload();
  auto data = payload->get_data();
  auto len = payload->get_length();

  if (msg->get_service() == GPS_SERVICE && len >= 8) {
    float lat, lon;
    std::memcpy(&lat, data, 4);
    std::memcpy(&lon, data + 4, 4);
    std::cout << "[CONSUMER] GPS  lat=" << lat << "  lon=" << lon << "\n";
  } else if (msg->get_service() == IMU_SERVICE && len >= 24) {
    float vals[6];
    std::memcpy(vals, data, 24);
    std::cout << "[CONSUMER] IMU  accel=(" << vals[0] << "," << vals[1] << ","
              << vals[2] << ")  gyro=(" << vals[3] << "," << vals[4] << ","
              << vals[5] << ")\n";
  }
}

void on_availability(vsomeip::service_t svc, vsomeip::instance_t inst,
                     bool available) {
  std::cout << "[CONSUMER] Serviço 0x" << std::hex << svc << " instância 0x"
            << inst << (available ? " DISPONIVEL" : " INDISPONIVEL") << std::dec
            << "\n";

  if (available) {
    if (svc == GPS_SERVICE)
      app->subscribe(GPS_SERVICE, INSTANCE_ID, EVTGRP_GPS);
    if (svc == IMU_SERVICE)
      app->subscribe(IMU_SERVICE, INSTANCE_ID, EVTGRP_IMU);
  }
}

int main() {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  app = vsomeip::runtime::get()->create_application("Consumer");
  if (!app->init()) {
    std::cerr << "[CONSUMER] Falha ao inicializar vSomeIP\n";
    return 1;
  }

  app->register_message_handler(vsomeip::ANY_SERVICE, vsomeip::ANY_INSTANCE,
                                vsomeip::ANY_METHOD, on_message);

  app->register_availability_handler(GPS_SERVICE, INSTANCE_ID, on_availability);
  app->register_availability_handler(IMU_SERVICE, INSTANCE_ID, on_availability);

  app->request_service(GPS_SERVICE, INSTANCE_ID);
  app->request_service(IMU_SERVICE, INSTANCE_ID);

  app->request_event(GPS_SERVICE, INSTANCE_ID, EVENT_GPS, {EVTGRP_GPS},
                     vsomeip::event_type_e::ET_FIELD);
  app->request_event(IMU_SERVICE, INSTANCE_ID, EVENT_IMU, {EVTGRP_IMU},
                     vsomeip::event_type_e::ET_FIELD);

  std::cout << "[CONSUMER] Aguardando serviços GPS e IMU...\n";
  app->start();
  return 0;
}
