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

#include "hyperbrain/conversion/bftollvm/BFToLLVM.h"
#include "hyperbrain/parser/BFParser.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    llvm::errs() << "usage: hyperbrain-cli <input-file>\n";
    std::exit(1);
  }

  auto filename = argv[1];
  auto input_res = hyperbrain::parser::Input::FromFile(filename);
  if (!input_res) {
    llvm::errs() << "input error: " << input_res.takeError() << "\n";
    std::exit(1);
  }

  auto input = *input_res;

  mlir::MLIRContext context;
  context.loadDialect<hyperbrain::bf::BFDialect>();

  hyperbrain::parser::BFTokenizer tokenizer(context);

  auto tokens_res = tokenizer.Tokenize(input);
  if (!tokens_res) {
    llvm::errs() << "tokenizer error: " << tokens_res.takeError() << "\n";
    std::exit(1);
  }

  auto tokens = *tokens_res;
  hyperbrain::parser::BFParser parser(context);

  auto err = parser.Parse(tokens);
  if (err) {
    llvm::errs() << "parser error: " << err << "\n";
    std::exit(1);
  }

  auto module = parser.Module();
  llvm::outs() << "==================== BF IR:\n";
  module.dump();

  mlir::PassManager manager(&context);
  manager.addPass(hyperbrain::conversion::createConvertBFToLLVMPass());
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());

  if (manager.run(module).failed()) {
    llvm::errs() << "failed to run passes\n";
    std::exit(1);
  }

  llvm::outs() << "==================== LLVM IR:\n";
  module.dump();

  mlir::registerLLVMDialectTranslation(context);
  mlir::registerBuiltinDialectTranslation(context);

  llvm::LLVMContext llvm_context;
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module.getOperation(), llvm_context);

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
  llvm_manager.run(*llvm_module, mam);

  llvm::outs() << "==================== Optimized LLVM IR:\n";
  llvm_module->dump();
}
