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
#include "hyperbrain/target/LLVMTarget.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

llvm::cl::OptionCategory HBCLICat("hyperbrain-cli options");

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"),
                                         llvm::cl::Required,
                                         llvm::cl::cat(HBCLICat));

llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("path for the output file"),
                   llvm::cl::init("a.out"), llvm::cl::cat(HBCLICat));

llvm::cl::opt<bool> OutputBF("b", llvm::cl::desc("compile to MLIR BF dialect"),
                             llvm::cl::init(false), llvm::cl::cat(HBCLICat));

llvm::cl::opt<bool>
    OutputLLVMIR("r", llvm::cl::desc("compile to MLIR LLVMIR dialect"),
                 llvm::cl::init(false), llvm::cl::cat(HBCLICat));

llvm::cl::opt<size_t>
    MemorySize("m", llvm::cl::desc("memory size for the program to allocate"),
               llvm::cl::init(1024), llvm::cl::cat(HBCLICat));

int main(int argc, char *argv[]) {
  llvm::cl::HideUnrelatedOptions(HBCLICat);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "HyperBrain CLI - High-performance BrainFuck compiler and runtime\n");

  auto input_res = hyperbrain::parser::Input::FromFile(InputFilename);
  if (!input_res) {
    llvm::errs() << "input error: " << input_res.takeError() << "\n";
    return 1;
  }
  auto input = *input_res;

  std::error_code ec;
  llvm::raw_fd_ostream os(OutputFilename, ec);
  if (ec) {
    llvm::errs() << "output error: " << ec.message() << "\n";
    return 1;
  }

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();

  mlir::MLIRContext context;
  hyperbrain::parser::BFTokenizer tokenizer(context);

  auto tokens_res = tokenizer.Tokenize(input);
  if (!tokens_res) {
    llvm::errs() << "tokenizer error: " << tokens_res.takeError() << "\n";
    return 1;
  }

  auto tokens = *tokens_res;
  hyperbrain::parser::BFParser parser(context);

  auto err = parser.Parse(tokens);
  if (err) {
    llvm::errs() << "parser error: " << err << "\n";
    return 1;
  }

  auto module = parser.Module();
  if (OutputBF) {
    module->print(os, flags);
    return 0;
  }

  mlir::PassManager manager(&context);
  hyperbrain::conversion::populateBFToLLVMPasses(manager);

  if (manager.run(module).failed()) {
    llvm::errs() << "failed to run passes\n";
    return 1;
  }

  if (OutputLLVMIR) {
    module->print(os, flags);
    return 0;
  }

  llvm::LLVMContext llvm_context;
  std::unique_ptr<llvm::Module> llvm_module =
      hyperbrain::target::translateToLLVM(module, llvm_context);
  hyperbrain::target::populateRuntimeFuncs(*llvm_module, MemorySize);
  hyperbrain::target::optimizeLLVMModule(*llvm_module);

  llvm_module->print(os, nullptr);
}
