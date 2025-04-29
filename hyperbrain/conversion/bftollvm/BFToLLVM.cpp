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

#include "hyperbrain/dialect/bf/BFOps.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace hyperbrain::conversion {

namespace {

struct ConvertBFNextOp : mlir::ConvertOpToLLVMPattern<bf::NextOp> {
  using ConvertOpToLLVMPattern<bf::NextOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::NextOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr = mlir::LLVM::LLVMPointerType::get(getContext());
    auto i8 = mlir::IntegerType::get(getContext(), 8);

    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        op, ptr, i8, adaptor.getP(), llvm::ArrayRef{mlir::LLVM::GEPArg(1)});
    return mlir::success();
  }
};

struct ConvertBFPrevOp : mlir::ConvertOpToLLVMPattern<bf::PrevOp> {
  using ConvertOpToLLVMPattern<bf::PrevOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::PrevOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr = mlir::LLVM::LLVMPointerType::get(getContext());
    auto i8 = mlir::IntegerType::get(getContext(), 8);

    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        op, ptr, i8, adaptor.getP(), llvm::ArrayRef{mlir::LLVM::GEPArg(-1)});
    return mlir::success();
  }
};

struct ConvertBFIncOp : mlir::ConvertOpToLLVMPattern<bf::IncOp> {
  using ConvertOpToLLVMPattern<bf::IncOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::IncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto i8 = mlir::IntegerType::get(getContext(), 8);

    auto val =
        rewriter.create<mlir::LLVM::LoadOp>(op->getLoc(), i8, adaptor.getP());
    auto one = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i8, 1);
    auto res = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), val, one);
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, res, adaptor.getP());
    return mlir::success();
  }
};

struct ConvertBFDecOp : mlir::ConvertOpToLLVMPattern<bf::DecOp> {
  using ConvertOpToLLVMPattern<bf::DecOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::DecOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto i8 = mlir::IntegerType::get(getContext(), 8);

    auto val =
        rewriter.create<mlir::LLVM::LoadOp>(op->getLoc(), i8, adaptor.getP());
    auto one = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), i8, 1);
    auto res = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), val, one);
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, res, adaptor.getP());
    return mlir::success();
  }
};

struct ConvertBFYieldOp : mlir::ConvertOpToLLVMPattern<bf::YieldOp> {
  using ConvertOpToLLVMPattern<bf::YieldOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto parent = op->getParentOp();
    if (llvm::isa<bf::WhileOp>(parent) ||
        llvm::isa<mlir::scf::WhileOp>(parent)) {
      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getP());
    } else if (llvm::isa<bf::MainOp>(parent) ||
               llvm::isa<mlir::LLVM::LLVMFuncOp>(parent)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, adaptor.getP());
    } else {
      return mlir::failure();
    }

    return mlir::success();
  }
};

struct ConvertBFWhileOp : mlir::ConvertOpToLLVMPattern<bf::WhileOp> {
  using ConvertOpToLLVMPattern<bf::WhileOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::WhileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ptr = mlir::LLVM::LLVMPointerType::get(getContext());
    auto i8 = mlir::IntegerType::get(getContext(), 8);
    auto loop =
        rewriter.create<mlir::scf::WhileOp>(op->getLoc(), ptr, adaptor.getP());

    auto cond = rewriter.createBlock(&loop.getBefore());
    auto p = cond->addArgument(ptr, op->getLoc());
    mlir::OpBuilder builder(cond, cond->begin());
    auto c = builder.create<mlir::LLVM::LoadOp>(op->getLoc(), i8, p);
    auto zero = builder.create<mlir::LLVM::ConstantOp>(op->getLoc(), i8, 0);
    auto r = builder.create<mlir::LLVM::ICmpOp>(
        op->getLoc(), mlir::LLVM::ICmpPredicate::ne, c, zero);
    builder.create<mlir::scf::ConditionOp>(op->getLoc(), r, p);

    loop.getAfter().takeBody(adaptor.getComp());
    if (failed(rewriter.convertRegionTypes(&loop.getAfter(),
                                           *getTypeConverter()))) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, loop);

    return mlir::success();
  }
};

struct ConvertBFMainOp : mlir::ConvertOpToLLVMPattern<bf::MainOp> {
  using ConvertOpToLLVMPattern<bf::MainOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::MainOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto name = LLVMBFMainFuncName;
    auto ptr = mlir::LLVM::LLVMPointerType::get(getContext());
    auto type = mlir::LLVM::LLVMFunctionType::get(ptr, {ptr});
    auto func =
        rewriter.create<mlir::LLVM::LLVMFuncOp>(op->getLoc(), name, type);

    func.getRegion().takeBody(adaptor.getComp());
    if (failed(rewriter.convertRegionTypes(&func.getRegion(),
                                           *getTypeConverter()))) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, func);

    return mlir::success();
  }
};

struct ConvertBFOutputOp : mlir::ConvertOpToLLVMPattern<bf::OutputOp> {
  using ConvertOpToLLVMPattern<bf::OutputOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::OutputOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto i8 = mlir::IntegerType::get(getContext(), 8);
    auto type = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(getContext()), {i8});
    auto name = LLVMBFPrintFuncName;

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module.lookupSymbol(name)) {
      auto ip = rewriter.saveInsertionPoint();
      auto block = &module.getRegion().front();
      rewriter.setInsertionPointToStart(block);

      rewriter.create<mlir::LLVM::LLVMFuncOp>(
          mlir::UnknownLoc::get(getContext()), name, type);

      rewriter.restoreInsertionPoint(ip);
    }

    auto c =
        rewriter.create<mlir::LLVM::LoadOp>(op->getLoc(), i8, adaptor.getP());
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, type, name,
                                                    mlir::ValueRange(c));

    return mlir::success();
  }
};

struct ConvertBFInputOp : mlir::ConvertOpToLLVMPattern<bf::InputOp> {
  using ConvertOpToLLVMPattern<bf::InputOp>::ConvertOpToLLVMPattern;
  mlir::LogicalResult
  matchAndRewrite(bf::InputOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto i8 = mlir::IntegerType::get(getContext(), 8);
    auto type = mlir::LLVM::LLVMFunctionType::get(i8, {});
    auto name = LLVMBFAcceptFuncName;

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module.lookupSymbol(name)) {
      auto ip = rewriter.saveInsertionPoint();
      auto block = &module.getRegion().front();
      rewriter.setInsertionPointToStart(block);

      rewriter.create<mlir::LLVM::LLVMFuncOp>(
          mlir::UnknownLoc::get(getContext()), name, type);

      rewriter.restoreInsertionPoint(ip);
    }

    auto c = rewriter.create<mlir::LLVM::CallOp>(op->getLoc(), type, name);
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, c->getResult(0),
                                                     adaptor.getP());

    return mlir::success();
  }
};

struct BFToLLVMConversionPass
    : public mlir::PassWrapper<BFToLLVMConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp module = getOperation();

    mlir::LLVMTypeConverter typeConverter(context);
    mlir::RewritePatternSet patterns(context);

    typeConverter.addConversion([this](bf::PtrType) {
      return mlir::LLVM::LLVMPointerType::get(&getContext());
    });

    conversion::populateBFToLLVMConversionPatterns(typeConverter, patterns);

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addIllegalDialect<bf::BFDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertBFToLLVMPass() {
  return std::make_unique<BFToLLVMConversionPass>();
}

void populateBFToLLVMConversionPatterns(mlir::LLVMTypeConverter &typeConverter,
                                        mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertBFMainOp, ConvertBFNextOp, ConvertBFPrevOp,
               ConvertBFIncOp, ConvertBFDecOp, ConvertBFYieldOp,
               ConvertBFWhileOp, ConvertBFInputOp, ConvertBFOutputOp>(
      typeConverter);
}

void populateBFToLLVMPasses(mlir::PassManager &pm) {
  pm.addPass(createConvertBFToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

} // namespace hyperbrain::conversion
