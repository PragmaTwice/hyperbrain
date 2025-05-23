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

#ifndef HYPERBRAIN_CONVERSION_BFTOLLVM_BFTOLLVM
#define HYPERBRAIN_CONVERSION_BFTOLLVM_BFTOLLVM

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"

namespace hyperbrain::conversion {

inline constexpr llvm::StringRef LLVMBFAcceptFuncName =
    "__hyperbrain_bf_accept";
inline constexpr llvm::StringRef LLVMBFPrintFuncName = "__hyperbrain_bf_print";
inline constexpr llvm::StringRef LLVMBFMainFuncName = "__hyperbrain_bf_main";

void populateBFToLLVMConversionPatterns(mlir::LLVMTypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createConvertBFToLLVMPass();

void populateCleanPasses(mlir::PassManager &pm);
void populateBFToLLVMPasses(mlir::PassManager &pm);

} // namespace hyperbrain::conversion

#endif
