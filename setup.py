#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt', 'r') as h:
    reqs = [p.strip() for p in h]

setup(name='olac',
      version='0.1.0',
      description='Online Learning at Cost',
      url='https://github.com/rurlus/olac',
      packages=['olac'],
      install_requires=reqs,
)
