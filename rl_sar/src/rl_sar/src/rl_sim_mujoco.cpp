/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"

#include <array>
#include <limits>
#include <sstream>

RL_Sim* RL_Sim::instance = nullptr;

namespace
{
struct Mid360Ray
{
    float theta;
    float phi;
};

struct Point3
{
    float x;
    float y;
    float z;
};

std::vector<Mid360Ray> LoadMid360Pattern()
{
    std::vector<Mid360Ray> rays;
    std::string pattern_path = std::string(POLICY_DIR) + "/g1/p3o_end2end/mid360_pattern_1024.csv";
    std::ifstream file(pattern_path);
    if (!file.is_open())
    {
        std::cout << LOGGER::ERROR << "Failed to open Mid360 pattern: " << pattern_path << std::endl;
        return rays;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string theta_str, phi_str;
        if (std::getline(ss, theta_str, ',') && std::getline(ss, phi_str, ','))
        {
            rays.push_back({std::stof(theta_str), std::stof(phi_str)});
        }
    }
    return rays;
}

std::vector<Point3> FpsSamplePointCloud(const std::vector<Point3>& points, const std::vector<bool>& valid, int num_samples)
{
    std::vector<Point3> sampled(num_samples, {0.0f, 0.0f, 0.0f});
    int valid_count = 0;
    for (bool is_valid : valid)
    {
        if (is_valid) valid_count++;
    }
    if (points.empty() || valid_count == 0 || num_samples <= 0)
    {
        return sampled;
    }

    float best_range = std::numeric_limits<float>::infinity();
    int current_idx = 0;
    for (size_t i = 0; i < points.size(); ++i)
    {
        if (!valid[i]) continue;
        float range_sq = points[i].x * points[i].x + points[i].y * points[i].y + points[i].z * points[i].z;
        if (range_sq < best_range)
        {
            best_range = range_sq;
            current_idx = static_cast<int>(i);
        }
    }

    std::vector<float> min_dist(points.size(), std::numeric_limits<float>::infinity());
    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx)
    {
        sampled[sample_idx] = sample_idx < valid_count ? points[current_idx] : Point3{0.0f, 0.0f, 0.0f};
        const Point3& current = points[current_idx];

        float farthest_dist = -1.0f;
        int farthest_idx = current_idx;
        for (size_t i = 0; i < points.size(); ++i)
        {
            float dx = points[i].x - current.x;
            float dy = points[i].y - current.y;
            float dz = points[i].z - current.z;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            min_dist[i] = std::min(min_dist[i], dist_sq);
            float masked_dist = valid[i] ? min_dist[i] : -1.0f;
            if (masked_dist > farthest_dist)
            {
                farthest_dist = masked_dist;
                farthest_idx = static_cast<int>(i);
            }
        }
        current_idx = farthest_idx;
    }
    return sampled;
}

float RaycastCylinder(const std::vector<float>& origin, const std::vector<float>& dir, const std::array<float, 3>& cylinder, float height, float max_distance)
{
    float ox = origin[0];
    float oy = origin[1];
    float oz = origin[2];
    float dx = dir[0];
    float dy = dir[1];
    float dz = dir[2];
    float cx = cylinder[0];
    float cy = cylinder[1];
    float radius = cylinder[2];

    float a = dx * dx + dy * dy;
    float b = 2.0f * ((ox - cx) * dx + (oy - cy) * dy);
    float c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f || a <= 1.0e-8f)
    {
        return max_distance;
    }

    float sqrt_disc = std::sqrt(discriminant);
    float inv_2a = 1.0f / (2.0f * a);
    float t1 = (-b - sqrt_disc) * inv_2a;
    float t2 = (-b + sqrt_disc) * inv_2a;
    float best = std::numeric_limits<float>::infinity();
    for (float t : {t1, t2})
    {
        float z = oz + t * dz;
        if (t > 0.0f && z >= 0.0f && z <= height)
        {
            best = std::min(best, t);
        }
    }
    if (!std::isfinite(best))
    {
        return max_distance;
    }
    return std::min(best, max_distance);
}
} // namespace

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        state->imu.quaternion[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 0];
        state->imu.quaternion[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 1];
        state->imu.quaternion[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 2];
        state->imu.quaternion[3] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 3];

        state->imu.gyroscope[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 4];
        state->imu.gyroscope[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 5];
        state->imu.gyroscope[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 6];

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            state->motor_state.q[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            state->motor_state.dq[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")];
            state->motor_state.tau_est[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + 2 * this->params.Get<int>("num_of_dofs")];
        }
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
        }
    }
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;
        if (this->params.Get<int>("lidar_num_points", 0) > 0)
        {
            int num_points = this->params.Get<int>("lidar_num_points", 128);
            float max_distance = this->params.Get<float>("lidar_points_clip", 6.0f);
            this->obs.lidar_points.resize(num_points * 3);

            static const std::vector<Mid360Ray> mid360_rays = LoadMid360Pattern();
            // Collect all p3o_obs_* obstacle cylinders
            struct CylinderObs { std::array<float,3> pos_radius; float height; };
            std::vector<CylinderObs> obstacles;
            for (int gi = 0; gi < this->mj_model->ngeom; ++gi)
            {
                const char* gname = mj_id2name(this->mj_model, mjOBJ_GEOM, gi);
                if (gname && std::string(gname).find("p3o_obs_") == 0)
                {
                    const mjtNum *gpos = &this->mj_data->geom_xpos[3 * gi];
                    const mjtNum *gsize = &this->mj_model->geom_size[3 * gi];
                    obstacles.push_back({{static_cast<float>(gpos[0]), static_cast<float>(gpos[1]), static_cast<float>(gsize[0])}, static_cast<float>(2.0f * gsize[1])});
                }
            }
            // Fallback: also support legacy single p3o_obstacle name
            if (obstacles.empty())
            {
                int legacy_id = mj_name2id(this->mj_model, mjOBJ_GEOM, "p3o_obstacle");
                if (legacy_id >= 0)
                {
                    const mjtNum *gpos = &this->mj_data->geom_xpos[3 * legacy_id];
                    const mjtNum *gsize = &this->mj_model->geom_size[3 * legacy_id];
                    obstacles.push_back({{static_cast<float>(gpos[0]), static_cast<float>(gpos[1]), static_cast<float>(gsize[0])}, static_cast<float>(2.0f * gsize[1])});
                }
            }
            int pelvis_body_id = mj_name2id(this->mj_model, mjOBJ_BODY, "pelvis");
            if (!obstacles.empty() && pelvis_body_id >= 0 && !mid360_rays.empty())
            {
                const mjtNum *base_pos = &this->mj_data->xpos[3 * pelvis_body_id];
                std::vector<float> sensor_offset = {0.10f, 0.0f, 0.63f};
                std::vector<float> sensor_offset_world = QuaternionMultiply(
                    QuaternionMultiply(this->obs.base_quat, {0.0f, sensor_offset[0], sensor_offset[1], sensor_offset[2]}),
                    QuaternionConjugate(this->obs.base_quat)
                );
                std::vector<float> sensor_origin_w = {
                    static_cast<float>(base_pos[0]) + sensor_offset_world[1],
                    static_cast<float>(base_pos[1]) + sensor_offset_world[2],
                    static_cast<float>(base_pos[2]) + sensor_offset_world[3]
                };

                std::vector<Point3> points_base;
                std::vector<bool> valid;
                points_base.reserve(mid360_rays.size());
                valid.reserve(mid360_rays.size());
                float half_fov = static_cast<float>(M_PI) * 0.5f;
                for (const Mid360Ray& ray : mid360_rays)
                {
                    std::vector<float> local_dir = {
                        std::cos(ray.theta) * std::cos(ray.phi),
                        std::sin(ray.theta) * std::cos(ray.phi),
                        std::sin(ray.phi)
                    };
                    std::vector<float> dir_world_quat = QuaternionMultiply(
                        QuaternionMultiply(this->obs.base_quat, {0.0f, local_dir[0], local_dir[1], local_dir[2]}),
                        QuaternionConjugate(this->obs.base_quat)
                    );
                    std::vector<float> dir_world = {dir_world_quat[1], dir_world_quat[2], dir_world_quat[3]};
                    // Raycast against all obstacles, keep closest hit
                    float hit_distance = max_distance;
                    for (const auto& obs : obstacles)
                    {
                        float d = RaycastCylinder(sensor_origin_w, dir_world, obs.pos_radius, obs.height, max_distance);
                        hit_distance = std::min(hit_distance, d);
                    }
                    Point3 point = {
                        local_dir[0] * hit_distance + sensor_offset[0],
                        local_dir[1] * hit_distance + sensor_offset[1],
                        local_dir[2] * hit_distance + sensor_offset[2]
                    };

                    float planar_range = std::sqrt(std::max(point.x * point.x + point.y * point.y, 1.0e-9f));
                    float theta = std::atan2(point.y, point.x);
                    bool point_valid =
                        point.x >= -0.5f && point.x <= 6.0f &&
                        std::abs(point.y) <= 3.0f &&
                        point.z >= -1.0f && point.z <= 0.8f &&
                        planar_range >= 0.2f && planar_range <= max_distance &&
                        std::abs(theta) <= half_fov;
                    points_base.push_back(point);
                    valid.push_back(point_valid);
                }

                std::vector<Point3> sampled = FpsSamplePointCloud(points_base, valid, num_points);
                for (int i = 0; i < num_points; ++i)
                {
                    this->obs.lidar_points[3 * i + 0] = sampled[i].x;
                    this->obs.lidar_points[3 * i + 1] = sampled[i].y;
                    this->obs.lidar_points[3 * i + 2] = sampled[i].z;
                }
            }
            else
            {
                for (int i = 0; i < num_points; ++i)
                {
                    this->obs.lidar_points[3 * i + 0] = 0.0f;
                    this->obs.lidar_points[3 * i + 1] = 0.0f;
                    this->obs.lidar_points[3 * i + 2] = 0.0f;
                }
            }
        }

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}
