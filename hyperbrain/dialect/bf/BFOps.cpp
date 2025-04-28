// Copyright 2025 PragmaTwice
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hyperbrain/dialect/bf/BFOps.h"

#include "hyperbrain/dialect/bf/BFOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hyperbrain/dialect/bf/BFOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "hyperbrain/dialect/bf/BFOps.cpp.inc"

namespace hyperbrain::bf {

void BFDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hyperbrain/dialect/bf/BFOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "hyperbrain/dialect/bf/BFOps.cpp.inc"
      >();
}

} // namespace hyperbrain::bf
