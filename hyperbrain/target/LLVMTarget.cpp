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

#include "hyperbrain/target/LLVMTarget.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Passes/PassBuilder.h"

namespace hyperbrain::target {

std::unique_ptr<llvm::Module> translateToLLVM(mlir::ModuleOp module,
                                              llvm::LLVMContext &ctx) {
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerBuiltinDialectTranslation(*module.getContext());

  return mlir::translateModuleToLLVMIR(module.getOperation(), ctx);
}

void optimizeLLVMModule(llvm::Module &module) {
  llvm::PassBuilder builder;

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  builder.registerModuleAnalyses(mam);
  builder.registerCGSCCAnalyses(cgam);
  builder.registerFunctionAnalyses(fam);
  builder.registerLoopAnalyses(lam);
  builder.crossRegisterProxies(lam, fam, cgam, mam);

  auto llvm_manager =
      builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  llvm_manager.run(module, mam);
}

} // namespace hyperbrain::target
