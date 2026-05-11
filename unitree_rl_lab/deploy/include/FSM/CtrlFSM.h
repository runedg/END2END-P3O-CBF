// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include "param.h"

class CtrlFSM
{
public:
    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));

    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            FSMStringMap.insert({id, fsm_name});
        }

        // Initialize FSM states
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            auto fsm_class = getFsmMap().find("State_" + fsm_type);
            if (fsm_class == getFsmMap().end()) {
                throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
            }
            auto state_instance = fsm_class->second(id, fsm_name);
            add(state_instance);
        }
    }

    void start() 
    {
        std::string init_state = "Passive";
        if (param::config["runtime"] && param::config["runtime"]["auto_start"])
        {
            init_state = param::config["runtime"]["auto_start"].as<std::string>();
        }

        if (states.empty())
        {
            throw std::runtime_error("FSM: no states registered");
        }

        auto pick_state = [&](const std::string& target)->std::shared_ptr<BaseState>
        {
            for (auto & state : states)
            {
                if (state && state->getStateString() == target)
                {
                    return state;
                }
            }
            return nullptr;
        };

        currentState = pick_state(init_state);
        if (!currentState)
        {
            currentState = pick_state("Passive");
        }
        if (!currentState)
        {
            for (auto & state : states)
            {
                if (state)
                {
                    currentState = state;
                    break;
                }
            }
        }
        if (!currentState)
        {
            throw std::runtime_error("FSM: all registered states are null");
        }
        currentState->enter();
        state_enter_time_ = std::chrono::steady_clock::now();

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }
    
    ~CtrlFSM()
    {
        states.clear();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    const double dt = 0.001;

    void run_()
    {
        currentState->pre_run();
        currentState->run();
        currentState->post_run();
        
        // Check if need to change state
        int nextStateMode = 0;
        for(int i(0); i<currentState->registered_checks.size(); i++)
        {
            if(currentState->registered_checks[i].first())
            {
                nextStateMode = currentState->registered_checks[i].second;
                break;
            }
        }

        if(nextStateMode == 0
            && currentState->getStateString() == "FixStand"
            && param::config["runtime"]
            && param::config["runtime"]["auto_velocity_after"]
            && FSMStringMap.right.count("Velocity"))
        {
            const auto delay_s = param::config["runtime"]["auto_velocity_after"].as<float>();
            const auto elapsed = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - state_enter_time_
            ).count();
            if (elapsed >= delay_s)
            {
                nextStateMode = FSMStringMap.right.at("Velocity");
            }
        }

        if(nextStateMode != 0 && !currentState->isState(nextStateMode))
        {
            for(auto & state : states)
            {
                if(state->isState(nextStateMode))
                {
                    spdlog::info("FSM: Change state from {} to {}", currentState->getStateString(), state->getStateString());
                    currentState->exit();
                    currentState = state;
                    currentState->enter();
                    state_enter_time_ = std::chrono::steady_clock::now();
                    break;
                }
            }
        }
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
    std::chrono::steady_clock::time_point state_enter_time_{};
};
