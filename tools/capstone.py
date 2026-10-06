#!/usr/bin/env python3
"""Local-first launcher; remote mode runs the same lifecycle on an existing host."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(CAPSTONE_MODE='local', CAPSTONE_GPU='0', CAPSTONE_VIEWER='0',
                CAPSTONE_VIEWER_PORT='6080', CAPSTONE_ROS_DOMAIN_ID='41',
                CAPSTONE_PROJECT='bar_cup_arrangement', CAPSTONE_CONTAINER='bar_cup_arrangement_backend',
                CAPSTONE_REMOTE_HOST='', CAPSTONE_REMOTE_WORKSPACE='')
ACTIONS = {'build-image':'up_container_build', 'build-workspace':'build_ws',
           'up':'bringup_dual', 'status':'status', 'down':'safe_down'}


def configuration(path=None, environ=None):
    values = DEFAULTS.copy()
    path = Path(path) if path else ROOT / '.hosting.env'
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, separator, value = line.partition('=')
            if not separator or key.strip() not in DEFAULTS:
                raise ValueError(f'Unsupported hosting setting at line {number}')
            values[key.strip()] = value.strip()
    environment = os.environ if environ is None else environ
    values.update({k: environment[k] for k in DEFAULTS if k in environment})
    if values['CAPSTONE_MODE'] not in ('local','remote'):
        raise ValueError('CAPSTONE_MODE must be local or remote')
    for key in ('CAPSTONE_GPU','CAPSTONE_VIEWER'):
        if values[key] not in ('0','1'):
            raise ValueError(key + ' must be 0 or 1')
    for key, minimum, maximum in [('CAPSTONE_VIEWER_PORT',1,65535),('CAPSTONE_ROS_DOMAIN_ID',0,232)]:
        if not values[key].isdigit() or not minimum <= int(values[key]) <= maximum:
            raise ValueError('Invalid ' + key)
    for key in ('CAPSTONE_PROJECT','CAPSTONE_CONTAINER'):
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',values[key]):
            raise ValueError('Invalid ' + key)
    if values['CAPSTONE_MODE']=='remote':
        host=values['CAPSTONE_REMOTE_HOST'];workspace=values['CAPSTONE_REMOTE_WORKSPACE']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@:-]*',host):
            raise ValueError('Set CAPSTONE_REMOTE_HOST to an existing authorized SSH host')
        if not PurePosixPath(workspace).is_absolute() or '\n' in workspace or '\r' in workspace:
            raise ValueError('Set an absolute CAPSTONE_REMOTE_WORKSPACE path')
    return values


def backend_environment(config):
    return dict(ENABLE_GPU=config['CAPSTONE_GPU'], SIM_HEADLESS='1',
                ENABLE_DESKTOP=config['CAPSTONE_VIEWER'], ENABLE_NOVNC=config['CAPSTONE_VIEWER'],
                NOVNC_HOST_PORT=config['CAPSTONE_VIEWER_PORT'], ROS_DOMAIN_ID=config['CAPSTONE_ROS_DOMAIN_ID'],
                COMPOSE_PROJECT_NAME=config['CAPSTONE_PROJECT'], CONTAINER=config['CAPSTONE_CONTAINER'],
                IGN_PARTITION=config['CAPSTONE_PROJECT'], PAUSE_LEGACY_BACKEND='0')


def command(config, action, root=ROOT):
    if action not in ACTIONS and action!='doctor':
        raise ValueError('Unsupported lifecycle action')
    env=backend_environment(config)
    if action=='doctor':
        operation=['docker','compose','version']
    else:
        operation=['bash','ros_backend1.1/scripts/backend11_lifecycle.sh',ACTIONS[action]]
    if config['CAPSTONE_MODE']=='local':
        return operation, Path(root), dict(os.environ,**env)
    remote='cd '+shlex.quote(config['CAPSTONE_REMOTE_WORKSPACE'])+' && '+shlex.join(['env',*[k+'='+v for k,v in env.items()],*operation])
    return ['ssh',config['CAPSTONE_REMOTE_HOST'],shlex.join(['bash','-lc',remote])],Path(root),None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['config','doctor',*ACTIONS])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    config=configuration(args.config)
    if args.action=='config':
        print(json.dumps(config,indent=2));return
    argv,cwd,env=command(config,args.action)
    if args.dry_run:
        print(json.dumps(dict(command=argv,cwd=str(cwd),backend_environment=backend_environment(config)),indent=2));return
    subprocess.run(argv,cwd=cwd,env=env,check=True)


if __name__=='__main__':main()
