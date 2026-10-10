"""Static road export: geometry, references, input isolation and nuPlan adapter."""
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import export_nuplan_bev_v85 as exporter
from bev_renderer_v85 import BevConfig, SEMANTIC_INDEX, render_bev, world_to_ego
from raw_record_bev_v85 import STATIC_INPUT_MODE
from static_road_planner_v85 import RoadScene, RoadGraph, RoadPlanningConfig, task_cost, hermite_motion


def arguments(*extra):
    return exporter.parse_args(["--data-root", "/unused", "--map-root", "/unused",
                               "--db-files", "unused.db", "--output-root", "/unused", *extra])


def lane(key="lane:a", y=0., xmin=-10., xmax=34., outgoing=()):
    return dict(id=key, kind="lane", baseline=[[xmin,y], [xmax,y]],
                polygon=[[xmin,y-3.5], [xmax,y-3.5], [xmax,y+3.5], [xmin,y+3.5]],
                outgoing=list(outgoing))


def road_scene(lanes=None, agents=(), hazards=(), **settings):
    return RoadScene(lanes or [lane()], list(agents), list(hazards), [-12,36,-24,24],
                     dict(length=4.8, width=2., rear_axle_to_center=1.4), RoadPlanningConfig(**settings))


def make_scene():
    names = ("LANE", "LANE_CONNECTOR", "CROSSWALK", "STOP_LINE",
             "ROADBLOCK", "ROADBLOCK_CONNECTOR", "DRIVABLE_AREA")
    layers = SimpleNamespace(**{name: name for name in names})
    primitive = lane()
    map_lane = SimpleNamespace(id="a", outgoing_edges=[], adjacent_edges=(None, None), polygon=SimpleNamespace(exterior=SimpleNamespace(
        coords=primitive["polygon"])), baseline_path=SimpleNamespace(linestring=SimpleNamespace(coords=primitive["baseline"])))

    def query(point, radius, requested):
        assert "DRIVABLE_AREA" not in requested
        return {layers.LANE: [map_lane]}

    def forbidden(*args, **kwargs):
        raise AssertionError("static export must not access route, history, traffic or future GT")

    def actor(x, y, kind, length, width):
        return SimpleNamespace(center=SimpleNamespace(x=x, y=y, heading=0.),
                               box=SimpleNamespace(length=length, width=width),
                               tracked_object_type=SimpleNamespace(fullname=kind))

    scene = SimpleNamespace(
        map_api=SimpleNamespace(get_proximal_map_objects=query),
        get_ego_state_at_iteration=lambda _: SimpleNamespace(
            rear_axle=SimpleNamespace(x=0., y=0., heading=0.), center=SimpleNamespace(x=1.4, y=0.)),
        get_tracked_objects_at_iteration=lambda _: [actor(12, 2.5, "pedestrian", .6, .6),
                                                   actor(20, 3.4, "traffic_cone", .5, .5)],
        get_route_roadblock_ids=forbidden, get_past_tracked_objects=forbidden,
        get_ego_future_trajectory=forbidden, get_traffic_light_status_at_iteration=forbidden,
        get_time_point=lambda _: SimpleNamespace(time_us=1000000),
        ego_vehicle_parameters=SimpleNamespace(length=4.8, width=2.),
        log_name="log", token="scene", scenario_type="test")
    state_module = ModuleType("nuplan.common.actor_state.state_representation")
    state_module.Point2D = lambda x, y: SimpleNamespace(x=x, y=y)
    map_module = ModuleType("nuplan.common.maps.maps_datatypes")
    map_module.SemanticMapLayer = layers
    return scene, {state_module.__name__: state_module, map_module.__name__: map_module}


BASE_TASK = dict(objective="length", forbid_hazards=False, pedestrian_weight=0.)


class StaticRoadTest(unittest.TestCase):
    def test_global_coordinates_keep_sub_meter_precision(self):
        origin = np.array([500000., 4200000., 0.])
        actual = world_to_ego([[500000.125,4200000.075]], origin)
        np.testing.assert_allclose(actual, [[.125,.075]], atol=1e-6)

    def test_default_geometry_and_legacy_opt_in(self):
        args = arguments()
        self.assertEqual((args.image_size, args.x_min, args.x_max, args.y_min, args.y_max),
                         (1440, -24., 72., -48., 48.))
        self.assertEqual(args.history_seconds, 0.)
        legacy = arguments("--planning-mode", "route_tracking")
        self.assertEqual((legacy.image_size, legacy.x_min, legacy.x_max, legacy.history_seconds), (288,-16.,64.,2.))

    def test_no_text_route_history_or_signals_in_model_image(self):
        line = np.array([[0,0], [25,0]])
        with patch.object(ImageDraw.ImageDraw, "text", side_effect=AssertionError("no text")):
            result = render_bev({"route": [lane()["polygon"]]}, {"reference": [line]}, [], [line],
                                {"red": [line]}, np.c_[line, [0,0]], [25,0], static_scene=True)
        for name in ("route", "reference", "history", "traffic_red"):
            self.assertFalse(result["bev_semantic"][SEMANTIC_INDEX[name]].any())

    def test_full_export_references_and_input_separation(self):
        from evaluate_static_bev_v85 import evaluate_record, model_inputs
        scene, modules = make_scene()
        args = arguments("--goal-xy", "26", "0", "--image-size", "360")
        config = BevConfig(image_size=360, x_min=-12,x_max=36,y_min=-24,y_max=24,history_seconds=0,history_samples=0)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules):
            self.assertTrue(exporter.export_iteration(scene, 0, directory, config, args))
            folder = Path(directory)/"log/scene/0000"
            self.assertEqual(len(list(folder.glob("*.npz"))), 3)
            inputs = []
            statuses = []
            for name in ("shortest", "pedestrian_clearance", "avoid_hazards"):
                path = folder/f"{name}.npz"
                with np.load(path, allow_pickle=False) as record:
                    self.assertEqual(record["schema_revision"].item(), 6)
                    self.assertFalse(record["trajectory_gt_available"].item())
                    self.assertNotIn("reference_path_ego", record)
                    self.assertNotIn("expert_trajectory_ego", record)
                    self.assertNotIn("planning_free", record)
                    np.testing.assert_array_equal(record["goal_xy"], [26,0])
                    np.testing.assert_array_equal(record["bev_rgb"], np.asarray(Image.open(folder/"bev_preview.png")))
                    inputs.append(model_inputs(record)["bev_rgb"])
                    self.assertNotIn("reference_trajectory_ego", model_inputs(record))
                    status = str(record["reference_status"].item())
                    statuses.append(status)
                    if status == "optimal_on_graph":
                        report = evaluate_record(path, record["reference_trajectory_ego"])
                        self.assertTrue(report["success"], report)
                        self.assertAlmostEqual(report["cost"], float(record["reference_cost"]), places=6)
                    else:
                        self.assertEqual(record["reference_trajectory_ego"].shape, (0,3))
            np.testing.assert_array_equal(inputs[0], inputs[1])
            np.testing.assert_array_equal(inputs[1], inputs[2])
            self.assertEqual(statuses[:2], ["optimal_on_graph"]*2)
            self.assertEqual(statuses[2], "infeasible_on_graph")
            with Image.open(folder/"bev_preview.png") as preview:
                self.assertEqual(preview.size, (360,360))

    def test_goal_is_forward_and_repeatable(self):
        graph = RoadGraph(road_scene(lateral_offsets_m=(0.,))).build()
        distances, _ = graph.solve(BASE_TASK)
        args = arguments()
        target = graph.choose_goal(distances, args, "a")
        self.assertEqual(target, graph.choose_goal(distances, args, "a"))
        self.assertGreaterEqual(graph.poses[target][0], args.goal_min_forward_m)
        self.assertLessEqual(np.linalg.norm(graph.poses[target][:2]), args.goal_max_distance)
        args.goal_xy = [-6,0]
        graph = RoadGraph(road_scene(lateral_offsets_m=(0.,))).build(args.goal_xy)
        self.assertIsNone(graph.choose_goal(graph.solve(BASE_TASK)[0], args, "a"))

    def test_explicit_bad_goal_is_not_replaced(self):
        scene, modules = make_scene()
        config = BevConfig(image_size=180,x_min=-12,x_max=36,y_min=-24,y_max=24)
        for xy in ((80,0), (-20,0), (24,20)):
            with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules):
                self.assertFalse(exporter.export_iteration(scene,0,directory,config,arguments("--goal-xy",*map(str,xy))))
                self.assertFalse(list(Path(directory).rglob("*.npz")))

    def test_swept_body_hits_between_waypoints_and_body_corner(self):
        obstacle = dict(center=[6,1.1],heading=0,length=.8,width=.6,type="barrier")
        scene = road_scene(agents=[obstacle])
        result = scene.assess(np.array([[0,0,0], [12,0,0]]))
        self.assertEqual(result["reason"], "body_collision")
        # Rear axle stays on the road, but the car hangs outside it.
        self.assertEqual(road_scene().assess([[0,3,0],[4,3,0]])["reason"], "body_outside_road_or_viewport")

    def test_reverse_sideways_and_tight_turn_are_rejected(self):
        scene = road_scene()
        self.assertFalse(scene.assess([[4,0,0],[0,0,0]])["feasible"])
        self.assertFalse(scene.assess([[0,0,0],[0,4,0]])["feasible"])
        self.assertIsNone(hermite_motion([0,0,0],[1,1,np.pi/2],.25,5.5))
        self.assertEqual(scene.assess([[0,0,np.pi],[1,0,np.pi]])["reason"], "reverse_or_lateral_motion")

    def test_no_proximity_jump_but_explicit_outgoing_is_traversable(self):
        first = lane(xmax=10.)
        second = lane(key="connector:b",xmin=10.,xmax=34.)
        second["kind"] = "connector"
        for connected in (False,True):
            first["outgoing"] = ["connector:b"] if connected else []
            graph = RoadGraph(road_scene([first,second],lateral_offsets_m=(0.,))).build([26,0])
            distance,_ = graph.solve(BASE_TASK)
            reachable = any(np.isfinite(distance[i]) for i in graph.explicit_nodes)
            self.assertEqual(reachable, connected)
        wrong = lane()
        wrong["baseline"] = wrong["baseline"][::-1]
        graph = RoadGraph(road_scene([wrong],lateral_offsets_m=(0.,))).build()
        self.assertFalse(graph.adjacency[0])

    def test_task_cost_changes_reference_and_dijkstra_matches_straight_distance(self):
        ped = dict(center=[12,2],heading=0,length=.8,width=.6,type="pedestrian")
        graph = RoadGraph(road_scene(agents=[ped])).build([26,0])
        target = graph.explicit_nodes[0]
        shortest = graph.reference(target, graph.solve(BASE_TASK), BASE_TASK)
        self.assertAlmostEqual(shortest["cost"], 26., places=6)
        task = dict(BASE_TASK,pedestrian_weight=4.)
        safer = graph.reference(target,graph.solve(task),task)
        self.assertLess(safer["cost"], task_cost(shortest["metrics"],task)-.1)
        self.assertLess(np.min(safer["poses"][:,1]), -.1)

    def test_lane_change_requires_adjacency_and_forward_curved_motion(self):
        lanes = [lane(), lane(key="lane:b", y=3.5)]
        for item, y in zip(lanes, (0.,3.5)):
            item["polygon"] = [[-10,y-1.75],[34,y-1.75],[34,y+1.75],[-10,y+1.75]]
        for adjacent in (False, True):
            lanes[0]["adjacent"] = ["lane:b"] if adjacent else []
            lanes[1]["adjacent"] = ["lane:a"] if adjacent else []
            graph = RoadGraph(road_scene(lanes,lateral_offsets_m=(0.,))).build([26,3.5])
            solution = graph.solve(BASE_TASK)
            target = graph.explicit_nodes[0]
            self.assertEqual(np.isfinite(solution[0][target]), adjacent)
            if adjacent:
                reference = graph.reference(target,solution,BASE_TASK)
                self.assertGreater(reference["cost"], 26.)
                self.assertLessEqual(reference["metrics"]["max_curvature_inv_m"], 1/5.5+1e-5)

    def test_legacy_processor_rejects_static_records(self):
        from planning_data_bev_v85 import process_bev_raw_record
        with self.assertRaisesRegex(ValueError,"static_multitask records"):
            process_bev_raw_record({"input_mode":np.str_(STATIC_INPUT_MODE)})


if __name__ == "__main__":
    unittest.main()
