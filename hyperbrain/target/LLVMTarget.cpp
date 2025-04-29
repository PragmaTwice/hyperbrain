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
#include "hyperbrain/conversion/bftollvm/BFToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Alignment.h"

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

void populateRuntimeFuncs(llvm::Module &module, size_t memory_size) {
  populateBFAcceptFunc(module);
  populateBFPrintFunc(module);
  populateMainFunc(module, memory_size);
}

void populateBFAcceptFunc(llvm::Module &module) {
  auto &ctx = module.getContext();
  llvm::IRBuilder<> builder(ctx);

  auto *getchar_type = llvm::FunctionType::get(builder.getInt32Ty(), false);
  auto getchar = module.getOrInsertFunction("getchar", getchar_type);

  auto *type = llvm::FunctionType::get(builder.getInt8Ty(), {}, false);

  auto *func = module.getFunction(conversion::LLVMBFAcceptFuncName);
  if (!func) {
    func = llvm::Function::Create(type, llvm::Function::ExternalLinkage,
                                  conversion::LLVMBFAcceptFuncName, module);
  }

  auto *entry = llvm::BasicBlock::Create(ctx, "entry", func);
  builder.SetInsertPoint(entry);

  auto *call = builder.CreateCall(getchar);
  auto *trunc = builder.CreateTrunc(call, builder.getInt8Ty());
  builder.CreateRet(trunc);

  llvm::verifyFunction(*func);
}

void populateBFPrintFunc(llvm::Module &module) {
  auto &ctx = module.getContext();
  llvm::IRBuilder<> builder(ctx);

  auto *putchar_type = llvm::FunctionType::get(builder.getInt32Ty(),
                                               {builder.getInt32Ty()}, false);
  auto putchar = module.getOrInsertFunction("putchar", putchar_type);

  auto *type = llvm::FunctionType::get(builder.getVoidTy(),
                                       {builder.getInt8Ty()}, false);

  auto *func = module.getFunction(conversion::LLVMBFPrintFuncName);
  if (!func) {
    func = llvm::Function::Create(type, llvm::Function::ExternalLinkage,
                                  conversion::LLVMBFPrintFuncName, module);
  }

  auto *entry = llvm::BasicBlock::Create(ctx, "entry", func);
  builder.SetInsertPoint(entry);

  auto *sext = builder.CreateSExt(func->getArg(0), builder.getInt32Ty());
  builder.CreateCall(putchar, {sext});
  builder.CreateRetVoid();

  llvm::verifyFunction(*func);
}

void populateMainFunc(llvm::Module &module, size_t memory_size) {
  auto &ctx = module.getContext();
  llvm::IRBuilder<> builder(ctx);

  auto *malloc_type = llvm::FunctionType::get(builder.getPtrTy(),
                                              {builder.getInt64Ty()}, false);
  auto malloc = module.getOrInsertFunction("malloc", malloc_type);

  auto *bfmain_type =
      llvm::FunctionType::get(builder.getPtrTy(), {builder.getPtrTy()}, false);
  auto bfmain =
      module.getOrInsertFunction(conversion::LLVMBFMainFuncName, bfmain_type);

  auto *type = llvm::FunctionType::get(builder.getInt32Ty(), {}, false);

  auto *func = llvm::Function::Create(type, llvm::Function::ExternalLinkage,
                                      "main", module);

  auto *entry = llvm::BasicBlock::Create(ctx, "entry", func);
  builder.SetInsertPoint(entry);

  auto size = builder.getInt64(memory_size);
  auto *ptr = builder.CreateCall(malloc, {size});
  builder.CreateMemSetInline(ptr, llvm::MaybeAlign(), builder.getInt8(0), size);
  builder.CreateCall(bfmain, {ptr});
  builder.CreateRet(builder.getInt32(0));

  llvm::verifyFunction(*func);
}

} // namespace hyperbrain::target
