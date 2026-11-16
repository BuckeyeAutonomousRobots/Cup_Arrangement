"""Exercise the actual probe query block without ROS or robot commands."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

tree=ast.parse(Path(__file__).with_name('startup_probe_rpc_retry.py').read_text())
block=next(n for n in tree.body if isinstance(n,ast.For))
code=compile(ast.Module(body=[block],type_ignores=[]),'probe-query-block','exec')

class Future:
    def __init__(self,ready): self.ready=ready; self.cancelled_flag=False
    def done(self):return self.ready
    def cancelled(self):return self.cancelled_flag
    def cancel(self):self.cancelled_flag=True
    def result(self):return NS(controller=[NS(name='controller',state='active')])

class Tests(unittest.TestCase):
    def run_case(self,responses,available=True):
        futures=[];waits=[];destroyed=[]
        class Client:
            def wait_for_service(self,timeout_sec):return available
            def call_async(self,request):
                f=Future(responses[len(futures)]);futures.append(f);return f
        node=NS(create_client=lambda *a:Client(),destroy_client=lambda c:destroyed.append(c))
        env=dict(n=node,controllers={},ListControllers=NS(Request=lambda:None),json=json,
                 rclpy=NS(spin_until_future_complete=lambda n,f,timeout_sec:waits.append(timeout_sec)))
        error=None
        with contextlib.redirect_stdout(io.StringIO()):
            try:exec(code,env)
            except AssertionError as exc:error=str(exc)
        return futures,waits,destroyed,env,error
    def test_immediate_success(self):
        f,w,d,e,err=self.run_case([True,True]);self.assertIsNone(err);self.assertEqual(len(f),2);self.assertEqual(len(d),2)
    def test_first_timeout_then_success(self):
        f,w,d,e,err=self.run_case([False,True,True]);self.assertIsNone(err);self.assertTrue(f[0].cancelled());self.assertEqual(w,[5,5,5])
    def test_both_timeout_fails_closed(self):
        f,w,d,e,err=self.run_case([False,False]);self.assertIn('after two bounded',err);self.assertEqual(len(f),2);self.assertTrue(all(x.cancelled() for x in f));self.assertEqual(e['controllers'],{})
    def test_missing_service_fails_closed(self):
        f,w,d,e,err=self.run_case([],False);self.assertIn('manager missing',err);self.assertEqual(f,[])

if __name__=='__main__':unittest.main()
