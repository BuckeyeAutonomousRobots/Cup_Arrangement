"""Regression for an unanswered start service leaving the bridge stuck forever."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from teleop_bridge.servo_bridge.servo_command_bridge import TargetTwistToServoCmd


def test_unanswered_start_request_retries():
    old, new = Mock(), Mock()
    old.done.return_value = False
    client = Mock()
    client.service_is_ready.return_value = True
    client.call_async.return_value = new
    bridge = SimpleNamespace(_start_client=client, _servo_started=False,
        _start_future=old, _start_request_time=0.0, _start_request_timeout_sec=5.0,
        _start_servo_service='/right_arm/servo_node/start_servo',
        get_logger=Mock(return_value=Mock()), _on_start_servo_done=Mock())
    with patch('teleop_bridge.servo_bridge.servo_command_bridge.time.monotonic', return_value=10.0):
        TargetTwistToServoCmd._request_start_servo(bridge, force=False, reason='test')
    old.cancel.assert_called_once()
    client.call_async.assert_called_once()
    assert bridge._start_future is new
    assert bridge._start_request_time == 10.0


def test_cancelled_start_response_is_ignored():
    future = Mock()
    future.cancelled.return_value = True
    TargetTwistToServoCmd._on_start_servo_done(SimpleNamespace(), future, 'test')
    future.result.assert_not_called()
