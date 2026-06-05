# Copyright 2023 The Langfun Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup for pip package."""

import collections
import datetime
import sys
from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
  """Gets current version of langfun package."""
  with open('langfun/__init__.py') as fp:
    version = None
    for line in fp:
      if line.startswith('__version__'):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        version = g['__version__']
        break
  if version is None:
    raise ValueError('`__version__` not defined in `langfun/__init__.py`')
  if '--nightly' in sys.argv:
    nightly_label = datetime.datetime.now().strftime('%Y%m%d%H%M')
    version = f'{version}.dev{nightly_label}'
    sys.argv.remove('--nightly')
  return version


def _parse_requirements(
    requirements_txt_path: str
) -> tuple[list[str], dict[str, list[str]]]:
  """Parses the require and extras_require for setup() from requirements.txt."""

  # Currently a requirements.txt is being used to specify dependencies. In order
  # to avoid specifying it in two places, we're going to use that file as the
  # source of truth.
  extras = collections.defaultdict(list)
  paths = ['require']

  def add_requirement(requirement: str, extra_key: str) -> None:
    if requirement not in extras[extra_key]:
      extras[extra_key].append(requirement)

  with open(requirements_txt_path) as file:
    for line in file:
      line = line.strip()
      if not line:
        continue

      if line.startswith('# extras:'):
        extra_path = line[line.find(':') + 1:].split('-')
        paths = [
            '-'.join(extra_path[:i + 1]) for i in range(len(extra_path))
        ]
      else:
        requirement, *_ = line.split('#')
        if requirement:
          for p in paths:
            add_requirement(requirement, p)
          add_requirement(requirement, 'all')

  require = extras.pop('require')
  return require, dict(extras)


_VERSION = _get_version()

install_requires, extras_require = _parse_requirements('requirements.txt')

setup(
    name='langfun',
    version=_VERSION,
    url='https://github.com/google/langfun',
    license='Apache License 2.0',
    author='Langfun Authors',
    description='Langfun: Language as Functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='langfun-authors@google.com',
    # Contained modules and scripts.
    packages=find_namespace_packages(include=['langfun*']),
    install_requires=install_requires,
    extras_require=extras_require,
    requires_python='>=3.10',
    include_package_data=True,
    # PyPI package information.
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    keywords=(
        'llm generative-ai machine-learning '
        ),
)
