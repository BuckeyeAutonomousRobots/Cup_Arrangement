import threading
import unittest
from async_sink import AsyncSink

class Tests(unittest.TestCase):
    def test_order_and_drain(self):
        rows=[]; s=AsyncSink(rows.append,100)
        for i in range(50): s.submit(i)
        stats=s.close()
        self.assertEqual(rows,list(range(50)))
        self.assertEqual(stats['written'],50)
    def test_write_error_rejected(self):
        def fail(item): raise OSError('disk failure')
        s=AsyncSink(fail,10);s.submit(1)
        with self.assertRaises(RuntimeError): s.close()
    def test_overflow_rejected(self):
        started=threading.Event(); release=threading.Event()
        def slow(item): started.set(); release.wait(5)
        s=AsyncSink(slow,1);s.submit(1);self.assertTrue(started.wait(2));s.submit(2)
        with self.assertRaises(RuntimeError):s.submit(3)
        release.set()
        with self.assertRaises(RuntimeError):s.close()

if __name__=='__main__':unittest.main()
