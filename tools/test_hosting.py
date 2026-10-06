import tempfile
import unittest
from pathlib import Path
from capstone import configuration, command, backend_environment


class HostingTests(unittest.TestCase):
    def config(self,**values):
        return configuration('/nonexistent/capstone-test-config',values)

    def test_local_default_has_no_remote_dependency(self):
        config=self.config()
        argv,cwd,env=command(config,'up',Path('/tmp/example-repository'))
        self.assertEqual(config['CAPSTONE_MODE'],'local')
        self.assertEqual(argv,['bash','ros_backend1.1/scripts/backend11_lifecycle.sh','bringup_dual'])
        self.assertEqual(cwd,Path('/tmp/example-repository'))
        self.assertEqual(env['ENABLE_GPU'],'0')
        self.assertEqual(env['ENABLE_DESKTOP'],'0')
        self.assertEqual(env['PAUSE_LEGACY_BACKEND'],'0')

    def test_same_lifecycle_in_remote_mode(self):
        config=self.config(CAPSTONE_MODE='remote',CAPSTONE_REMOTE_HOST='user@example.org',
                           CAPSTONE_REMOTE_WORKSPACE='/srv/Cup Arrangement',CAPSTONE_GPU='1')
        argv,_,_=command(config,'up')
        self.assertEqual(argv[:2],['ssh','user@example.org'])
        self.assertIn('bringup_dual',argv[2]);self.assertIn('ENABLE_GPU=1',argv[2])
        self.assertNotIn('scp',argv[2])

    def test_environment_overrides_file_without_shell_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'hosting.env';p.write_text('CAPSTONE_GPU=0\nCAPSTONE_PROJECT=test_project\n')
            config=configuration(p,{'CAPSTONE_GPU':'1'})
            self.assertEqual(config['CAPSTONE_GPU'],'1')
            self.assertEqual(config['CAPSTONE_PROJECT'],'test_project')

    def test_invalid_remote_and_network_settings_rejected(self):
        for values in [dict(CAPSTONE_MODE='remote'),dict(CAPSTONE_VIEWER_PORT='70000'),
                       dict(CAPSTONE_ROS_DOMAIN_ID='999'),dict(CAPSTONE_MODE='remote',
                       CAPSTONE_REMOTE_HOST='-oProxyCommand=bad',CAPSTONE_REMOTE_WORKSPACE='/srv/project')]:
            with self.assertRaises(ValueError):self.config(**values)

    def test_viewer_and_domain_are_explicit(self):
        env=backend_environment(self.config(CAPSTONE_VIEWER='1',CAPSTONE_VIEWER_PORT='6085',CAPSTONE_ROS_DOMAIN_ID='42'))
        self.assertEqual(env['ENABLE_NOVNC'],'1');self.assertEqual(env['NOVNC_HOST_PORT'],'6085')
        self.assertEqual(env['ROS_DOMAIN_ID'],'42')


if __name__=='__main__':unittest.main()
