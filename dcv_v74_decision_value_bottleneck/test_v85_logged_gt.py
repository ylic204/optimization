"""GT boundary crop: first intersection, real times, exact goal and no leakage."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from bev_renderer_v85 import BevConfig
import export_nuplan_bev_v85 as exporter
from evaluate_static_bev_v85 import model_inputs
from logged_gt_goal_v85 import boundary_inset, geometric_gt, load_logged_future, truncate_at_boundary
from static_road_planner_v85 import RoadGraph
from test_v85_static_bev import arguments, make_scene, road_scene


class LoggedGTTest(unittest.TestCase):
    def test_first_boundary_intersection_preserves_prefix_times_and_stops(self):
        poses = np.array([[0,0,0],[0,0,0],[6,0,0],[14,0,.4],[5,0,0]])
        times = np.array([0.,1.,4.,8.,12.])
        result = truncate_at_boundary(poses,times,[-10,10,-10,10],2.)
        np.testing.assert_array_equal(result['poses'][:3],poses[:3])
        np.testing.assert_allclose(result['poses'][-1],[8,0,.1])
        np.testing.assert_allclose(result['times'],[0,1,4,5])
        self.assertEqual(len(geometric_gt(result['poses'])),3)
        self.assertTrue(result['end_interpolated'])

    def test_side_exit_and_shortest_angle_interpolation(self):
        poses = [[0,0,0],[2,6,3.1],[2,10,-3.1]]
        result = truncate_at_boundary(poses,[0,1,2],[-10,10,-10,10],2.)
        np.testing.assert_allclose(result['poses'][-1,:2],[2,8])
        self.assertAlmostEqual(abs(result['poses'][-1,2]),np.pi)
        self.assertAlmostEqual(result['times'][-1],1.5)
        self.assertIsNone(truncate_at_boundary([[0,0,0],[4,0,0]],[0,1],[-10,10,-10,10],2.))

    def test_invalid_timing_is_rejected(self):
        with self.assertRaises(ValueError):
            truncate_at_boundary([[0,0,0],[14,0,0]],[0,0],[-10,10,-10,10],2.)

    def test_actual_timestamp_and_global_ego_transform(self):
        source = SimpleNamespace(
            get_time_point=lambda _:SimpleNamespace(time_us=1000000),
            get_ego_future_trajectory=lambda *a,**k: [
                SimpleNamespace(time_point=SimpleNamespace(time_us=t),rear_axle=SimpleNamespace(x=500000+x,y=4200000.,heading=0.))
                for t,x in ((1700000,.125),(2350000,.25))])
        poses,times = load_logged_future(source,0,[500000,4200000,0],20,.1)
        np.testing.assert_allclose(times,[0,.7,1.35])
        np.testing.assert_allclose(poses[:,0],[0,.125,.25])

    def test_exact_gt_goal_is_not_snapped_to_baseline_or_heading(self):
        graph = RoadGraph(road_scene(lateral_offsets_m=(0.,))).build()
        requested = np.array([26.,.2,.07])
        target = graph.add_pose_goal(requested)
        np.testing.assert_array_equal(graph.poses[target],requested)
        blocked = graph.add_pose_goal([27.,15.,.1])
        self.assertFalse(np.isfinite(graph.solve(dict(objective='length',forbid_hazards=False,pedestrian_weight=0.))[0][blocked]))

    def test_gt_export_beyond_old_goal_range_and_numeric_reference(self):
        scene,modules = make_scene()
        query = scene.map_api.get_proximal_map_objects
        def extended_map(*args):
            values = query(*args)
            lane = values['LANE'][0]
            lane.polygon.exterior.coords=[[-20,-3.5],[120,-3.5],[120,3.5],[-20,3.5]]
            lane.baseline_path.linestring.coords=[[-20,0],[120,0]]
            return values
        scene.map_api.get_proximal_map_objects=extended_map
        scene.get_ego_future_trajectory=lambda *a,**k: [
            SimpleNamespace(time_point=SimpleNamespace(time_us=1000000+int(t*1e6)),
                            rear_axle=SimpleNamespace(x=5*t,y=0.,heading=0.))
            for t in np.arange(.1,20.01,.1)]
        config=BevConfig(image_size=360,x_min=-24,x_max=72,y_min=-48,y_max=48,history_seconds=0,history_samples=0)
        for skip in (False,True):
            args=arguments(*(['--skip-numeric-reference'] if skip else []))
            with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',modules):
                self.assertTrue(exporter.export_iteration(scene,0,directory,config,args))
                folder=Path(directory)/'log/scene/0000'
                self.assertTrue((folder/'audit/logged_gt.png').exists())
                self.assertEqual((folder/'audit/planning_graph.npz').exists(),not skip)
                with np.load(folder/'shortest.npz',allow_pickle=False) as record:
                    self.assertTrue(record['trajectory_gt_available'])
                    self.assertFalse(record['logged_gt_is_task_optimum'])
                    self.assertEqual(record['goal_source'].item(),'logged_gt_boundary')
                    np.testing.assert_allclose(record['logged_gt_trajectory_ego'][-1],record['goal_pose'])
                    self.assertGreater(record['goal_xy'][0],28.)
                    self.assertAlmostEqual(record['goal_xy'][0]+record['logged_gt_boundary_inset_m'],72.,places=5)
                    self.assertAlmostEqual(record['logged_gt_time_s'][-1],record['goal_pose'][0]/5,places=5)
                    self.assertNotIn('logged_gt_trajectory_ego',model_inputs(record))
                    self.assertEqual(record['reference_status'].item(),'not_computed' if skip else 'optimal_on_graph')
                    self.assertFalse(record['bev_semantic'][4].any())
                    with Image.open(folder/'bev_preview.png') as image:
                        np.testing.assert_array_equal(np.asarray(image),record['bev_rgb'])

    def test_no_boundary_reached_skips_without_fake_goal(self):
        scene,modules=make_scene()
        scene.get_ego_future_trajectory=lambda *a,**k: [SimpleNamespace(
            time_point=SimpleNamespace(time_us=2000000),rear_axle=SimpleNamespace(x=2,y=0,heading=0))]
        args=arguments('--skip-numeric-reference')
        config=BevConfig(image_size=180,x_min=-24,x_max=72,y_min=-48,y_max=48)
        with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',modules):
            self.assertFalse(exporter.export_iteration(scene,0,directory,config,args))
            self.assertEqual(args.skip_reasons,{'logged_gt_does_not_reach_boundary':1})
            self.assertFalse(list(Path(directory).rglob('*.npz')))


if __name__=='__main__':
    unittest.main()
