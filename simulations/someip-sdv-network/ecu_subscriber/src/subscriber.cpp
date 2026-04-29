/**
 * @file subscriber.cpp
 * @brief ECU Subscriber genérico — usado por ADAS, CLU, NAV, PT, STE, TEL
 *
 * Comportamento determinado pela variável de ambiente ECU_NAME:
 *
 *   ECU_NAME  | Subscreve
 *   ----------|-------------------------------------------
 *   ADAS      | VDE (0x1236)
 *   CLU       | GPS (0x1234)
 *   NAV       | GPS (0x1234)
 *   PT        | VDE (0x1236)
 *   STE       | IMU (0x1235)
 *   TEL       | GPS (0x1234) + IMU (0x1235)
 */

#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <vsomeip/vsomeip.hpp>

// ── IDs dos serviços ──────────────────────────────────────────────────────────
static constexpr vsomeip::service_t    GPS_SVC   = 0x1234;
static constexpr vsomeip::service_t    IMU_SVC   = 0x1235;
static constexpr vsomeip::service_t    VDE_SVC   = 0x1236;
static constexpr vsomeip::instance_t   INST      = 0x0001;
static constexpr vsomeip::event_t      EVT_GPS   = 0x8001;
static constexpr vsomeip::event_t      EVT_IMU   = 0x8002;
static constexpr vsomeip::event_t      EVT_VDE   = 0x8003;
static constexpr vsomeip::eventgroup_t EVTG_GPS  = 0x0001;
static constexpr vsomeip::eventgroup_t EVTG_IMU  = 0x0002;
static constexpr vsomeip::eventgroup_t EVTG_VDE  = 0x0003;

struct Subscription {
    vsomeip::service_t    service;
    vsomeip::event_t      event;
    vsomeip::eventgroup_t evtgroup;
};

// Mapa ECU → lista de serviços a subscrever (Figura 2 do artigo)
static const std::map<std::string, std::vector<Subscription>> ECU_SUBSCRIPTIONS = {
    {"ADAS", {{ VDE_SVC, EVT_VDE, EVTG_VDE }}},
    {"CLU",  {{ GPS_SVC, EVT_GPS, EVTG_GPS }}},
    {"NAV",  {{ GPS_SVC, EVT_GPS, EVTG_GPS }}},
    {"PT",   {{ VDE_SVC, EVT_VDE, EVTG_VDE }}},
    {"STE",  {{ IMU_SVC, EVT_IMU, EVTG_IMU }}},
    {"TEL",  {{ GPS_SVC, EVT_GPS, EVTG_GPS }, { IMU_SVC, EVT_IMU, EVTG_IMU }}},
};

static std::atomic<bool>                     running{true};
static std::shared_ptr<vsomeip::application> app;
static std::string                           ecu_name;
static std::vector<Subscription>             my_subs;

void signal_handler(int) { running = false; app->stop(); }

void on_message(const std::shared_ptr<vsomeip::message>& msg)
{
    auto data = msg->get_payload()->get_data();
    auto len  = msg->get_payload()->get_length();
    auto svc  = msg->get_service();

    if (svc == GPS_SVC && len >= 8) {
        float lat, lon;
        std::memcpy(&lat, data,     4);
        std::memcpy(&lon, data + 4, 4);
        std::cout << "[" << ecu_name << "] GPS  lat=" << lat << "  lon=" << lon << "\n";
    }
    else if (svc == IMU_SVC && len >= 24) {
        float v[6]; std::memcpy(v, data, 24);
        std::cout << "[" << ecu_name << "] IMU  accel=("
                  << v[0] << "," << v[1] << "," << v[2] << ")  gyro=("
                  << v[3] << "," << v[4] << "," << v[5] << ")\n";
    }
    else if (svc == VDE_SVC && len >= 16) {
        float v[4]; std::memcpy(v, data, 16);
        std::cout << "[" << ecu_name << "] VDE  speed=" << v[0]
                  << " km/h  steer=" << v[1] << "°  throttle=" << v[2]
                  << "%  brake=" << v[3] << "%\n";
    }
}

void on_availability(vsomeip::service_t svc, vsomeip::instance_t inst, bool available)
{
    std::cout << "[" << ecu_name << "] Serviço 0x" << std::hex << svc
              << (available ? " DISPONIVEL" : " INDISPONIVEL") << std::dec << "\n";

    if (!available) return;
    for (const auto& s : my_subs) {
        if (s.service == svc)
            app->subscribe(svc, inst, s.evtgroup);
    }
}

int main()
{
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    const char* env = std::getenv("ECU_NAME");
    ecu_name = env ? env : "UNKNOWN";

    auto it = ECU_SUBSCRIPTIONS.find(ecu_name);
    if (it == ECU_SUBSCRIPTIONS.end()) {
        std::cerr << "[SUBSCRIBER] ECU_NAME inválido: '" << ecu_name << "'\n";
        std::cerr << "  Valores válidos: ADAS CLU NAV PT STE TEL\n";
        return 1;
    }
    my_subs = it->second;

    app = vsomeip::runtime::get()->create_application(ecu_name);
    if (!app->init()) {
        std::cerr << "[" << ecu_name << "] Falha ao inicializar vSomeIP\n";
        return 1;
    }

    app->register_message_handler(vsomeip::ANY_SERVICE, vsomeip::ANY_INSTANCE,
                                  vsomeip::ANY_METHOD, on_message);

    // Registra handlers e solicita eventos para cada subscrição
    std::set<vsomeip::service_t> seen;
    for (const auto& s : my_subs) {
        if (!seen.count(s.service)) {
            app->register_availability_handler(s.service, INST, on_availability);
            app->request_service(s.service, INST);
            seen.insert(s.service);
        }
        app->request_event(s.service, INST, s.event, {s.evtgroup},
                           vsomeip::event_type_e::ET_FIELD);
    }

    std::cout << "[" << ecu_name << "] Aguardando serviços...\n";
    app->start();
    return 0;
}
