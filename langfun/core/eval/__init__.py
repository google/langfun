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
"""langfun eval framework."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order

from langfun.core.eval.base import register
from langfun.core.eval.base import registered_names
from langfun.core.eval.base import get_evaluations
from langfun.core.eval.base import get
from langfun.core.eval.base import run

from langfun.core.eval.base import Evaluable
from langfun.core.eval.base import Evaluation
from langfun.core.eval.base import Suite
from langfun.core.eval.base import Summary

from langfun.core.eval.base import load
from langfun.core.eval.base import monitor
from langfun.core.eval.base import monitor_async

# Functors for loading inputs.
from langfun.core.eval.base import inputs_from
from langfun.core.eval.base import as_inputs

from langfun.core.eval.matching import Matching
from langfun.core.eval.scoring import Scoring

# Experiment patching.
from langfun.core.eval.patching import patch_member

from langfun.core.eval.patching import patch_lm
from langfun.core.eval.patching import patch_parsing_lm
from langfun.core.eval.patching import patch_inputs
from langfun.core.eval.patching import patch_prompt
from langfun.core.eval.patching import patch_schema_fn

# Placeholder for Google-internal imports.
# pylint: enable=g-bad-import-order
# pylint: enable=g-importing-member
