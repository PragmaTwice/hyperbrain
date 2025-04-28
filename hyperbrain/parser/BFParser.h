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

#ifndef HYPERBRAIN_PARSER_BFPARSER
#define HYPERBRAIN_PARSER_BFPARSER

#include "hyperbrain/dialect/bf/BFOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace hyperbrain::parser {

struct Token {
  enum Kind : char {
    NEXT = '>',
    PREV = '<',
    INC = '+',
    DEC = '-',
    OUTPUT = '.',
    INPUT = ',',
    WHILE_BEGIN = '[',
    WHILE_END = ']',
  };

  Kind kind;
  mlir::Location loc;

  Token(Kind kind, mlir::Location loc) : kind(kind), loc(loc) {}

  Token(Kind kind, mlir::MLIRContext &ctx)
      : Token(kind, mlir::UnknownLoc::get(&ctx)) {}
  Token(Kind kind, llvm::StringRef filename, size_t line, size_t column,
        mlir::MLIRContext &ctx)
      : Token(kind, mlir::FileLineColLoc::get(&ctx, filename, line, column)) {}

  std::string ToString() {
    std::string l;
    llvm::raw_string_ostream os(l);
    loc.print(os);
    auto ch = std::string(1, (char)kind);

    return "Token(" + ch + ", " + os.str() + ")";
  }
};

struct Input {
  llvm::StringRef Filename() const { return filename; }
  llvm::StringRef Content() const { return content; }

  static Input FromMemory(llvm::StringRef content) {
    return Input("<memory>", content);
  }

  static llvm::Expected<Input> FromFile(llvm::StringRef filename) {
    auto result = llvm::MemoryBuffer::getFile(filename, true);
    if (!result)
      return llvm::createStringError(result.getError(), "failed to read file");

    return Input(filename, result->get()->getBuffer());
  }

private:
  std::string filename;
  std::string content;

  Input(llvm::StringRef filename, llvm::StringRef content)
      : filename(filename), content(content){};
};

class BFTokenizer {
public:
  using Result = std::vector<Token>;
  llvm::Expected<Result> Tokenize(const Input &input);

  BFTokenizer(mlir::MLIRContext &ctx) : ctx(ctx) {}

  static bool IsTokenChar(char c);

private:
  mlir::MLIRContext &ctx;
};

class BFParser {
public:
  using Input = BFTokenizer::Result;

  llvm::Error Parse(const Input &input);

  BFParser(mlir::MLIRContext &ctx)
      : module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx))),
        builder(module.getBodyRegion()) {
    ctx.loadDialect<bf::BFDialect>();
  }

  mlir::ModuleOp Module() { return module; }

private:
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
};

} // namespace hyperbrain::parser

#endif
