#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator
from .utils import TeleopEvents, make_teleoperator_from_config

# Register Omega teleoperators when available so their config types show up in the CLI.
try:  # pragma: no cover - optional dependency
    from .omega3 import ForceDimensionOmega, ForceDimensionOmegaConfig  # noqa: F401
except ImportError:  # pragma: no cover - SDK not installed
    pass
