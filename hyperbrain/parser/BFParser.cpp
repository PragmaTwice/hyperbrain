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

#include "BFParser.h"
#include "hyperbrain/dialect/bf/BFOps.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/Error.h"

namespace hyperbrain::parser {

bool BFTokenizer::IsTokenChar(char c) {
  return c == Token::NEXT || c == Token::PREV || c == Token::INC ||
         c == Token::DEC || c == Token::OUTPUT || c == Token::INPUT ||
         c == Token::WHILE_BEGIN || c == Token::WHILE_END;
}

llvm::Expected<BFTokenizer::Result> BFTokenizer::Tokenize(const Input &input) {
  BFTokenizer::Result result;

  auto filename = input.Filename();
  size_t line = 1;
  size_t column = 0;
  size_t paren_level = 0;

  for (char c : input.Content()) {
    if (IsTokenChar(c)) {
      if (c == Token::WHILE_BEGIN)
        paren_level++;
      else if (c == Token::WHILE_END)
        paren_level--;

      if (paren_level < 0)
        return llvm::createStringError("parenthesis not matched");

      result.emplace_back(Token::Kind(c), filename, line, column, ctx);

      column++;
    } else if (std::isspace(c)) {
      if (c == '\n') {
        line++;
        column = 0;
      } else {
        column++;
      }
    } else {
      return llvm::createStringError("encountered invalid character");
    }
  }

  if (paren_level != 0)
    return llvm::createStringError("parenthesis not matched");

  return result;
}

llvm::Error BFParser::Parse(const BFParser::Input &input) {
  auto unkown_loc = mlir::UnknownLoc::get(builder.getContext());
  auto main = builder.create<bf::MainOp>(unkown_loc);

  auto comp = &main.getComp();
  auto block = builder.createBlock(comp);
  auto ptr = bf::PtrType::get(builder.getContext());
  mlir::Value current_val = block->addArgument(ptr, unkown_loc);

  for (const auto &t : input) {
    if (t.kind == Token::NEXT) {
      current_val = builder.create<bf::NextOp>(t.loc, ptr, current_val);
    } else if (t.kind == Token::PREV) {
      current_val = builder.create<bf::PrevOp>(t.loc, ptr, current_val);
    } else if (t.kind == Token::INC) {
      builder.create<bf::IncOp>(t.loc, current_val);
    } else if (t.kind == Token::DEC) {
      builder.create<bf::DecOp>(t.loc, current_val);
    } else if (t.kind == Token::INPUT) {
      builder.create<bf::InputOp>(t.loc, current_val);
    } else if (t.kind == Token::OUTPUT) {
      builder.create<bf::OutputOp>(t.loc, current_val);
    } else if (t.kind == Token::WHILE_BEGIN) {
      auto val = builder.create<bf::WhileOp>(t.loc, ptr, current_val);
      auto block = builder.createBlock(&val.getRegion());
      current_val = block->addArgument(ptr, t.loc);
      builder.setInsertionPointToStart(block);
    } else if (t.kind == Token::WHILE_END) {
      builder.create<bf::YieldOp>(t.loc, current_val);
      current_val = current_val.getParentRegion()->getParentOp()->getResult(0);
      builder.setInsertionPointAfterValue(current_val);
    }
  }

  builder.create<bf::YieldOp>(unkown_loc, current_val);

  return llvm::Error::success();
}

} // namespace hyperbrain::parser
