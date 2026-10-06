#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/wait_for_message.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace
{
template <typename T>
T get_or_declare(const rclcpp::Node::SharedPtr& node, const std::string& name, const T& fallback)
{
  if (!node->has_parameter(name))
  {
    node->declare_parameter<T>(name, fallback);
  }
  return node->get_parameter(name).get_value<T>();
}

std::string vec3(const Eigen::Vector3d& v)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(4) << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return out.str();
}

std::string pair_key(const std::string& a, const std::string& b)
{
  return a <= b ? a + " <-> " + b : b + " <-> " + a;
}

struct DistanceRow
{
  std::string key;
  double distance;
  Eigen::Vector3d p0;
  Eigen::Vector3d p1;
};

void apply_joint_state(
  const moveit::core::RobotModelConstPtr& model,
  moveit::core::RobotState& state,
  const sensor_msgs::msg::JointState& joint_state)
{
  std::set<std::string> variables(model->getVariableNames().begin(), model->getVariableNames().end());
  std::size_t applied = 0;
  for (std::size_t i = 0; i < joint_state.name.size() && i < joint_state.position.size(); ++i)
  {
    const auto& name = joint_state.name[i];
    if (variables.find(name) == variables.end())
    {
      continue;
    }
    state.setVariablePosition(name, joint_state.position[i]);
    ++applied;
  }
  state.update();
  RCLCPP_INFO(rclcpp::get_logger("self_collision_inspector"), "Applied %zu joint variables from joint state", applied);
}

void print_contacts(const collision_detection::CollisionResult& result, int max_pairs)
{
  if (!result.collision)
  {
    RCLCPP_INFO(rclcpp::get_logger("self_collision_inspector"), "Self collision contacts: none");
    return;
  }

  int printed = 0;
  RCLCPP_WARN(
    rclcpp::get_logger("self_collision_inspector"),
    "Self collision contacts found: contact_count=%zu pairs=%zu",
    result.contact_count,
    result.contacts.size());

  for (const auto& item : result.contacts)
  {
    if (printed >= max_pairs)
    {
      break;
    }
    const auto& contacts = item.second;
    const auto& first = contacts.front();
    RCLCPP_WARN(
      rclcpp::get_logger("self_collision_inspector"),
      "  contact[%d] %s, depth=%.6f, pos=%s, normal=%s",
      printed + 1,
      pair_key(first.body_name_1, first.body_name_2).c_str(),
      first.depth,
      vec3(first.pos).c_str(),
      vec3(first.normal).c_str());
    ++printed;
  }
}

void print_distances(const collision_detection::DistanceResult& result, int max_pairs)
{
  std::vector<DistanceRow> rows;
  rows.reserve(result.distances.size());
  for (const auto& item : result.distances)
  {
    for (const auto& data : item.second)
    {
      DistanceRow row;
      row.key = pair_key(data.link_names[0], data.link_names[1]);
      row.distance = data.distance;
      row.p0 = data.nearest_points[0];
      row.p1 = data.nearest_points[1];
      rows.push_back(row);
    }
  }
  std::sort(rows.begin(), rows.end(), [](const DistanceRow& a, const DistanceRow& b) {
    return a.distance < b.distance;
  });

  const auto& min = result.minimum_distance;
  RCLCPP_INFO(
    rclcpp::get_logger("self_collision_inspector"),
    "Minimum self distance: %.6f between %s and %s, nearest=%s -> %s, collision=%s",
    min.distance,
    min.link_names[0].c_str(),
    min.link_names[1].c_str(),
    vec3(min.nearest_points[0]).c_str(),
    vec3(min.nearest_points[1]).c_str(),
    result.collision ? "true" : "false");

  if (rows.empty())
  {
    RCLCPP_INFO(rclcpp::get_logger("self_collision_inspector"), "Detailed distance rows: none");
    return;
  }

  const int count = std::min<int>(max_pairs, rows.size());
  RCLCPP_INFO(rclcpp::get_logger("self_collision_inspector"), "Closest detailed self-distance pairs:");
  for (int i = 0; i < count; ++i)
  {
    const auto& row = rows[i];
    RCLCPP_INFO(
      rclcpp::get_logger("self_collision_inspector"),
      "  distance[%d] %s = %.6f, nearest=%s -> %s",
      i + 1,
      row.key.c_str(),
      row.distance,
      vec3(row.p0).c_str(),
      vec3(row.p1).c_str());
  }
}

void inspect_scene(
  const planning_scene::PlanningScene& scene,
  const moveit::core::RobotState& state,
  const std::string& group_name,
  double distance_threshold,
  int max_pairs,
  bool use_unpadded)
{
  const auto& model = scene.getRobotModel();
  const std::string label = use_unpadded ? "unpadded" : "padded";

  collision_detection::CollisionRequest contact_req;
  contact_req.contacts = true;
  contact_req.max_contacts = static_cast<std::size_t>(std::max(1, max_pairs * 4));
  contact_req.max_contacts_per_pair = 4;
  contact_req.distance = true;
  contact_req.detailed_distance = true;
  contact_req.group_name = group_name;

  collision_detection::CollisionResult contact_result;
  if (use_unpadded)
  {
    scene.getCollisionEnvUnpadded()->checkSelfCollision(contact_req, contact_result, state, scene.getAllowedCollisionMatrix());
  }
  else
  {
    scene.getCollisionEnv()->checkSelfCollision(contact_req, contact_result, state, scene.getAllowedCollisionMatrix());
  }

  RCLCPP_INFO(
    rclcpp::get_logger("self_collision_inspector"),
    "=== %s self-collision check, group='%s' ===",
    label.c_str(),
    group_name.empty() ? "<whole robot>" : group_name.c_str());
  print_contacts(contact_result, max_pairs);

  collision_detection::DistanceRequest distance_req;
  distance_req.group_name = group_name;
  distance_req.enableGroup(model);
  distance_req.enable_nearest_points = true;
  distance_req.enable_signed_distance = true;
  distance_req.type = collision_detection::DistanceRequestTypes::ALL;
  distance_req.max_contacts_per_body = static_cast<std::size_t>(std::max(1, max_pairs));
  distance_req.distance_threshold = distance_threshold;
  distance_req.acm = &scene.getAllowedCollisionMatrix();

  collision_detection::DistanceResult distance_result;
  if (use_unpadded)
  {
    scene.getCollisionEnvUnpadded()->distanceSelf(distance_req, distance_result, state);
  }
  else
  {
    scene.getCollisionEnv()->distanceSelf(distance_req, distance_result, state);
  }
  print_distances(distance_result, max_pairs);
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared(
    "self_collision_inspector",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  const auto joint_states_topic = get_or_declare<std::string>(node, "joint_states_topic", "/joint_states");
  const auto group_name = get_or_declare<std::string>(node, "group_name", "");
  const double wait_timeout_sec = get_or_declare<double>(node, "wait_timeout_sec", 3.0);
  const double distance_threshold = get_or_declare<double>(node, "distance_threshold", 0.08);
  const int max_pairs = get_or_declare<int>(node, "max_pairs", 20);

  robot_model_loader::RobotModelLoader loader(node, "robot_description");
  const auto model = loader.getModel();
  if (!model)
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to load robot model from robot_description");
    rclcpp::shutdown();
    return 2;
  }

  planning_scene::PlanningScene scene(model);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();

  sensor_msgs::msg::JointState joint_state;
  const auto timeout = std::chrono::milliseconds(static_cast<int>(wait_timeout_sec * 1000.0));
  if (rclcpp::wait_for_message(joint_state, node, joint_states_topic, timeout))
  {
    RCLCPP_INFO(node->get_logger(), "Using joint state from %s", joint_states_topic.c_str());
    apply_joint_state(model, state, joint_state);
  }
  else
  {
    RCLCPP_WARN(
      node->get_logger(),
      "No joint state received on %s within %.2fs; inspecting default state",
      joint_states_topic.c_str(),
      wait_timeout_sec);
    state.update();
  }

  if (!group_name.empty() && !model->hasJointModelGroup(group_name))
  {
    RCLCPP_WARN(node->get_logger(), "Group '%s' does not exist; whole robot distance query will be used", group_name.c_str());
  }

  inspect_scene(scene, state, group_name, distance_threshold, max_pairs, false);
  inspect_scene(scene, state, group_name, distance_threshold, max_pairs, true);
  inspect_scene(scene, state, "", distance_threshold, max_pairs, false);
  inspect_scene(scene, state, "", distance_threshold, max_pairs, true);

  rclcpp::shutdown();
  return 0;
}
