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
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

namespace hyperbrain::target {

std::unique_ptr<llvm::Module> translateToLLVM(mlir::ModuleOp module,
                                              llvm::LLVMContext &ctx,
                                              llvm::StringRef name) {
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerBuiltinDialectTranslation(*module.getContext());

  return mlir::translateModuleToLLVMIR(module.getOperation(), ctx, name);
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
  auto *ptr = builder.CreateAlloca(builder.getInt8Ty(), size);
  builder.CreateMemSetInline(ptr, llvm::MaybeAlign(), builder.getInt8(0), size);
  builder.CreateCall(bfmain, {ptr});
  builder.CreateRet(builder.getInt32(0));

  llvm::verifyFunction(*func);
}

llvm::Error emitObjectFile(llvm::Module &module, llvm::raw_pwrite_stream &os) {
  using namespace llvm;

  std::string target_triple = sys::getDefaultTargetTriple();
  module.setTargetTriple(target_triple);

  std::string error;
  const Target *target = TargetRegistry::lookupTarget(target_triple, error);
  if (!target) {
    return make_error<StringError>("failed to lookup target: " + error,
                                   inconvertibleErrorCode());
  }

  TargetOptions opt;
  auto rm = std::optional<Reloc::Model>();
  std::unique_ptr<TargetMachine> target_machine(
      target->createTargetMachine(target_triple, "generic", "", opt, rm));
  if (!target_machine) {
    return make_error<StringError>("failed to create target machine",
                                   inconvertibleErrorCode());
  }

  module.setDataLayout(target_machine->createDataLayout());

  legacy::PassManager pass;
  if (target_machine->addPassesToEmitFile(pass, os, nullptr,
                                          CodeGenFileType::ObjectFile)) {
    return make_error<StringError>("failed to emit object file",
                                   inconvertibleErrorCode());
  }

  pass.run(module);
  os.flush();

  return Error::success();
}

} // namespace hyperbrain::target
