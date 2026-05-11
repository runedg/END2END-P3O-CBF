#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <algorithm>
#include <unordered_map>
#include <cmath>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    const float orientation_limit = param::config["runtime"] && param::config["runtime"]["orientation_limit"]
        ? param::config["runtime"]["orientation_limit"].as<float>()
        : 1.0f;

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&, orientation_limit]()->bool{
                const auto & g = env->robot->data.projected_gravity_b;
                const float cos_value = std::clamp(-g[2], -1.0f, 1.0f);
                const float angle = std::fabs(std::acos(cos_value));
                const bool bad = angle > orientation_limit;
                if (bad)
                {
                    spdlog::warn(
                        "RL termination: bad_orientation angle={:.3f} limit={:.3f} proj_g=[{:.3f}, {:.3f}, {:.3f}]",
                        angle, orientation_limit, g[0], g[1], g[2]
                    );
                }
                return bad;
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    static int debug_counter = 0;
    const bool debug_rl = param::config["runtime"] && param::config["runtime"]["debug_rl"]
        ? param::config["runtime"]["debug_rl"].as<bool>()
        : false;
    if (debug_rl && (++debug_counter % 500 == 0))
    {
        const auto & g = env->robot->data.projected_gravity_b;
        const float angle = std::fabs(std::acos(std::clamp(-g[2], -1.0f, 1.0f)));
        const auto & joy = env->robot->data.joystick;
        const auto action = env->action_manager->processed_actions();
        float mean_abs_action = 0.0f;
        float max_abs_action = 0.0f;
        for (int i = 0; i < action.size(); ++i)
        {
            const float abs_v = std::fabs(action[i]);
            mean_abs_action += abs_v;
            max_abs_action = std::max(max_abs_action, abs_v);
        }
        mean_abs_action /= std::max<int>(1, action.size());
        spdlog::info(
            "RL debug: angle={:.3f} joy=[ly:{:.3f}, lx:{:.3f}, rx:{:.3f}] gravity=[{:.3f}, {:.3f}, {:.3f}] action_mean_abs={:.3f} action_max_abs={:.3f}",
            angle, joy->ly(), joy->lx(), joy->rx(), g[0], g[1], g[2], mean_abs_action, max_abs_action
        );
    }
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
