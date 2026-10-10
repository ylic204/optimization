"""Atomic export, strict resume and one-time compatibility checks for old r7."""
from pathlib import Path
import tempfile
import json
from unittest.mock import patch
from types import SimpleNamespace
import unittest

import numpy as np

from bev_renderer_v85 import BevConfig
from resume_bev_export_v85 import export_static_frame, MARKER
from test_v85_static_bev import arguments, make_scene
import export_nuplan_bev_v85 as exporter


class ResumeTest(unittest.TestCase):
    def test_resume_skips_work_and_rejects_changed_parameters_or_missing_files(self):
        scene=SimpleNamespace(log_name="log",token="scene",get_time_point=lambda _:SimpleNamespace(time_us=123))
        cfg=BevConfig()
        calls=[]
        def export(s,i,root,c,a):
            calls.append(1)
            folder=Path(root)/"log/scene/0000"
            folder.mkdir(parents=True)
            np.savez_compressed(folder/"shortest.npz",x=np.arange(5))
            (folder/"audit.json").write_text('{"ok":true}')
            return True
        with tempfile.TemporaryDirectory() as directory:
            args=arguments()
            self.assertTrue(export_static_frame(scene,0,directory,cfg,args,export))
            folder=Path(directory)/"log/scene/0000"
            self.assertTrue((folder/MARKER).is_file())
            with self.assertRaises(FileExistsError):
                export_static_frame(scene,0,directory,cfg,args,export)
            args.resume=True
            self.assertIsNone(export_static_frame(scene,0,directory,cfg,args,export))
            self.assertEqual(len(calls),1)
            args.gt_horizon_s=40.
            with self.assertRaisesRegex(ValueError,"different export parameters"):
                export_static_frame(scene,0,directory,cfg,args,export)
            args.gt_horizon_s=20.
            (folder/"audit.json").unlink()
            with self.assertRaisesRegex(ValueError,"incomplete/incompatible"):
                export_static_frame(scene,0,directory,cfg,args,export)
            self.assertTrue((folder/"shortest.npz").exists())

    def test_legacy_frame_recomputed_once_and_incomplete_output_preserved(self):
        scene=SimpleNamespace(log_name="log",token="scene",get_time_point=lambda _:SimpleNamespace(time_us=123))
        cfg=BevConfig()
        calls=[]
        def export(s,i,root,c,a):
            calls.append(1)
            folder=Path(root)/"log/scene/0000"
            folder.mkdir(parents=True)
            np.savez_compressed(folder/"shortest.npz",x=np.arange(5))
            return True
        args=arguments("--resume")
        with tempfile.TemporaryDirectory() as directory:
            export(scene,0,directory,cfg,args)
            self.assertIsNone(export_static_frame(scene,0,directory,cfg,args,export))
            self.assertEqual(len(calls),2)
            self.assertIsNone(export_static_frame(scene,0,directory,cfg,args,export))
            self.assertEqual(len(calls),2)
            folder=Path(directory)/"log/scene/0000"
            (folder/MARKER).unlink()
            np.savez_compressed(folder/"shortest.npz",x=np.arange(8))
            with self.assertRaisesRegex(ValueError,"incomplete/incompatible"):
                export_static_frame(scene,0,directory,cfg,args,export)
            with np.load(folder/"shortest.npz") as record:
                self.assertEqual(len(record["x"]),8)
            self.assertFalse((folder/MARKER).exists())

    def test_main_deduplicates_frame_identity(self):
        scene, _ = make_scene()
        scene.get_number_of_iterations = lambda: 1
        with tempfile.TemporaryDirectory() as directory:
            args = arguments()
            args.output_root = directory
            with patch.object(exporter, "parse_args", return_value=args), patch.object(
                    exporter, "build_scenarios", return_value=[scene, scene]), patch.object(
                    exporter, "export_iteration", return_value=True) as write:
                exporter.main()
            self.assertEqual(write.call_count, 1)
            summary = json.loads((Path(directory)/"export_summary.json").read_text())
            self.assertEqual(summary["duplicate_frames"], 1)
            self.assertEqual(summary["written_scenes"], 1)

    def test_failed_export_never_publishes_partial_frame(self):
        scene=SimpleNamespace(log_name="log",token="scene",get_time_point=lambda _:SimpleNamespace(time_us=123))
        def export(s,i,root,c,a):
            folder=Path(root)/"log/scene/0000"
            folder.mkdir(parents=True)
            (folder/"partial.txt").write_text("partial")
            raise RuntimeError("interrupted")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,"interrupted"):
                export_static_frame(scene,0,directory,BevConfig(),arguments(),export)
            self.assertFalse((Path(directory)/"log/scene/0000").exists())
            self.assertEqual(list(Path(directory).iterdir()),[])


if __name__=="__main__":
    unittest.main()
