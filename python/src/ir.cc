#include "ir.h"

#include <optional>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/LocationSnapshot.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

void setAsyncTaskIds(mlir::Operation *op,
                     llvm::ArrayRef<AsyncTaskId> asyncTaskIds) {
  llvm::SmallVector<AsyncTaskId> sortedAsyncTaskIds(asyncTaskIds.begin(),
                                                    asyncTaskIds.end());
  sort(sortedAsyncTaskIds);
  auto i32Ty = IntegerType::get(op->getContext(), 32);
  auto size = static_cast<int64_t>(sortedAsyncTaskIds.size());
  auto vecTy = VectorType::get(size, i32Ty);
  op->setAttr("async_task_id",
              DenseI32ArrayAttr::get(op->getContext(), sortedAsyncTaskIds));
}

namespace {

namespace py = pybind11;
using namespace mlir;
using namespace triton;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

llvm::raw_fd_ostream &mlir_dumps() {
  std::error_code EC;
  static llvm::raw_fd_ostream S(::triton::tools::getStrEnv("MLIR_DUMP_PATH"),
                                EC, llvm::sys::fs::CD_CreateAlways);
  assert(!EC);
  return S;
}

llvm::raw_ostream &mlir_dumps_or_dbgs() {
  if (!::triton::tools::getStrEnv("MLIR_DUMP_PATH").empty()) {
    return mlir_dumps();
  } else {
    return llvm::dbgs();
  }
}

// Run the pass manager under a source manager diagnostic handler, which
// enables emitted MLIR diagnostics to directly reference Python source
// code. This diagnostic handler supports filtering diagnostic info by
// severity levels.
struct TritonSourceMgrDiagnosticHandler : public SourceMgrDiagnosticHandler {
  TritonSourceMgrDiagnosticHandler(MLIRContext *ctx,
                                   DiagnosticSeverity minSeverity)
      : SourceMgrDiagnosticHandler(sourceMgr, ctx, llvm::errs()) {
    setHandler([this, minSeverity](Diagnostic &diag) {
      auto severity = diag.getSeverity();
      switch (severity) {
      case DiagnosticSeverity::Error:
        break;
      case DiagnosticSeverity::Warning:
        if (minSeverity == DiagnosticSeverity::Error)
          return success();
        break;
      case DiagnosticSeverity::Remark:
        if (minSeverity == DiagnosticSeverity::Error ||
            minSeverity == DiagnosticSeverity::Warning)
          return success();
        break;
      case DiagnosticSeverity::Note:
        // notes are handled somewhere else.
        return failure();
      default:
        llvm_unreachable("Unknown diagnostic severity");
      }
      emitDiagnostic(diag);
      return success();
    });
  }

  llvm::SourceMgr sourceMgr;
};

std::string locationToString(Location loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  loc.print(os);
  os.flush(); // Make sure all the content is dumped into the 'str' string
  return str;
}

// Function to parse a comma-separated string into a vector of C-style strings
llvm::SmallVector<const char *, 3>
parseCommaSeparatedValues(const std::string &input,
                          llvm::SmallVector<std::string, 3> &storage) {
  llvm::SmallVector<StringRef, 3> split;
  llvm::SmallVector<const char *, 3> result;
  StringRef(input.c_str()).split(split, ',');
  llvm::transform(split, std::back_inserter(result), [&storage](StringRef str) {
    // StringRefs are not always null-terminated.
    // The purpose for this storage pattern is to
    // produce a collection of C-strings that are.
    storage.push_back(str.str());
    return storage.back().c_str();
  });
  return result;
}

void outputWarning(Location loc, const std::string &msg) {
  std::string locStr = locationToString(loc);

  PyErr_WarnEx(PyExc_UserWarning, (locStr + ": " + msg).c_str(),
               /*stack_level=*/2);
}

// Allow dump a reproducer in the console on crash.
struct ConsoleReproducerStream : public mlir::ReproducerStream {
  ~ConsoleReproducerStream() override {}

  StringRef description() override {
    return "std::errs, please share the reproducer above with Triton project.";
  }
  raw_ostream &os() override { return llvm::errs(); }
};

ReproducerStreamFactory makeConsoleReproducer() {
  return [](std::string &error) -> std::unique_ptr<ReproducerStream> {
    return std::make_unique<ConsoleReproducerStream>();
  };
}

OpPrintingFlags getOpPrintingFlags() {
  auto printingFlags = OpPrintingFlags();
  printingFlags.enableDebugInfo();
  return printingFlags;
}

py::list getTensorDescMetadata(ModuleOp &mod) {
  py::list result;
  triton::FuncOp kernelFunc;
  mod.walk([&](triton::FuncOp func) {
    if (triton::isKernel(func)) {
      kernelFunc = func;
      return WalkResult::interrupt();
    }
    return WalkResult::skip();
  });
  assert(kernelFunc);

  for (auto [i, argTy] : llvm::enumerate(kernelFunc.getArgumentTypes())) {
    auto descTy = dyn_cast<TensorDescType>(argTy);
    if (!descTy)
      continue;

    auto blockType = descTy.getBlockType();
    auto encoding = blockType.getEncoding();
    auto mmaEncoding = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(encoding);
    auto swizzle = ttng::getTMASwizzleMode(nullptr, descTy);
    auto elemType = ttng::getTMAElementType(nullptr, descTy);
    assert(swizzle.has_value());
    assert(elemType.has_value());
    auto blockSize = ttng::getTMABlockShape(blockType, /*packedSize=*/false);
    py::dict metadata;
    metadata["swizzle"] = *swizzle;
    metadata["elem_size"] = descTy.getBlockType().getElementTypeBitWidth() / 8;
    metadata["elem_type"] = *elemType;
    metadata["block_size"] =
        std::vector<int>(blockSize.begin(), blockSize.end());
    metadata["fp4_padded"] = mmaEncoding && mmaEncoding.getFp4Padded();
    result.append(std::move(metadata));
  }
  return result;
}

} // anonymous namespace

/*****************************************************************************/
/* Python bindings for ir                                                    */
/*****************************************************************************/

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::enum_<PaddingOption>(m, "PADDING_OPTION", py::module_local())
      .value("PAD_ZERO", PaddingOption::PAD_ZERO)
      .value("PAD_NAN", PaddingOption::PAD_NAN)
      .export_values();

  py::enum_<CacheModifier>(m, "CACHE_MODIFIER", py::module_local())
      .value("NONE", CacheModifier::NONE)
      .value("CA", CacheModifier::CA)
      .value("CG", CacheModifier::CG)
      .value("WB", CacheModifier::WB)
      .value("CS", CacheModifier::CS)
      .value("WT", CacheModifier::WT)
      .value("CV", CacheModifier::CV)
      .export_values();

  py::enum_<MemSemantic>(m, "MEM_SEMANTIC", py::module_local())
      .value("ACQUIRE_RELEASE", MemSemantic::ACQUIRE_RELEASE)
      .value("ACQUIRE", MemSemantic::ACQUIRE)
      .value("RELEASE", MemSemantic::RELEASE)
      .value("RELAXED", MemSemantic::RELAXED)
      .export_values();

  py::enum_<MemSyncScope>(m, "MEM_SYNC_SCOPE", py::module_local())
      .value("GPU", MemSyncScope::GPU)
      .value("CTA", MemSyncScope::CTA)
      .value("SYSTEM", MemSyncScope::SYSTEM)
      .export_values();

  py::enum_<EvictionPolicy>(m, "EVICTION_POLICY", py::module_local())
      .value("NORMAL", EvictionPolicy::NORMAL)
      .value("EVICT_FIRST", EvictionPolicy::EVICT_FIRST)
      .value("EVICT_LAST", EvictionPolicy::EVICT_LAST)
      .export_values();

  py::enum_<RMWOp>(m, "ATOMIC_OP", py::module_local())
      .value("ADD", RMWOp::ADD)
      .value("FADD", RMWOp::FADD)
      .value("AND", RMWOp::AND)
      .value("OR", RMWOp::OR)
      .value("XOR", RMWOp::XOR)
      .value("XCHG", RMWOp::XCHG)
      .value("MAX", RMWOp::MAX)
      .value("MIN", RMWOp::MIN)
      .value("UMIN", RMWOp::UMIN)
      .value("UMAX", RMWOp::UMAX);

  py::enum_<DescriptorReduceKind>(m, "DESCRIPTOR_REDUCE_KIND",
                                  py::module_local())
      .value("ADD", DescriptorReduceKind::ADD)
      .value("AND", DescriptorReduceKind::AND)
      .value("OR", DescriptorReduceKind::OR)
      .value("XOR", DescriptorReduceKind::XOR)
      .value("MAX", DescriptorReduceKind::MAX)
      .value("MIN", DescriptorReduceKind::MIN)
      .value("INC", DescriptorReduceKind::INC)
      .value("DEC", DescriptorReduceKind::DEC);

  py::enum_<RoundingMode>(m, "ROUNDING_MODE", py::module_local())
      .value("RTZ", RoundingMode::RTZ)
      .value("RTNE", RoundingMode::RTNE);

  py::enum_<PropagateNan>(m, "PROPAGATE_NAN", py::module_local())
      .value("NONE", PropagateNan::NONE)
      .value("ALL", PropagateNan::ALL);

  py::enum_<InputPrecision>(m, "INPUT_PRECISION", py::module_local())
      .value("TF32", InputPrecision::TF32)
      .value("TF32x3", InputPrecision::TF32x3)
      .value("IEEE", InputPrecision::IEEE)
      .export_values();

  py::enum_<ScaleDotElemType>(m, "ScaleDotElemTypeTY", py::module_local())
      .value("E4M3", ScaleDotElemType::E4M3)
      .value("E5M2", ScaleDotElemType::E5M2)
      .value("E2M3", ScaleDotElemType::E2M3)
      .value("E3M2", ScaleDotElemType::E3M2)
      .value("E2M1", ScaleDotElemType::E2M1)
      .value("BF16", ScaleDotElemType::BF16)
      .value("FP16", ScaleDotElemType::FP16)
      .export_values();

  py::class_<MLIRContext>(m, "context", py::module_local())
      .def(py::init<>())
      .def("printOpOnDiagnostic",
           [](MLIRContext &self, bool v) { self.printOpOnDiagnostic(v); })
      .def("printStackTraceOnDiagnostic",
           [](MLIRContext &self, bool v) {
             self.printStackTraceOnDiagnostic(v);
           })
      .def("disable_multithreading",
           [](MLIRContext &self) { self.disableMultithreading(); });

  py::class_<SourceMgrDiagnosticHandler>(m, "source_mgr_diag",
                                         py::module_local())
      .def(py::init<llvm::SourceMgr &, MLIRContext *>());

  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<TritonDialect, ::mlir::triton::gpu::TritonGPUDialect,
                    math::MathDialect, arith::ArithDialect, scf::SCFDialect,
                    ::mlir::gpu::GPUDialect, cf::ControlFlowDialect,
                    ::mlir::triton::proton::ProtonDialect, LLVM::LLVMDialect,
                    mlir::ub::UBDialect>();
    mlir::LLVM::registerInlinerInterface(registry);
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    mlir::LLVM::registerInlinerInterface(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<Type>(m, "type", py::module_local())
      .def("is_integer",
           [](Type &self, unsigned width) { return self.isInteger(width); })
      .def("is_fp16", &Type::isF16)
      .def("__eq__",
           [](Type &self, py::object &other) {
             Type *other_ty = py::cast<Type *>(other);
             return (other_ty != nullptr) && (*other_ty == self);
           })
      .def("__ne__",
           [](Type &self, py::object &other) {
             Type *other_ty = py::cast<Type *>(other);
             return (other_ty == nullptr) || (*other_ty != self);
           })
      .def("__str__", [](Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<FunctionType>(m, "function_type", py::module_local())
      .def("param_types", [](FunctionType &self) {
        return std::vector<Type>(self.getInputs().begin(),
                                 self.getInputs().end());
      });

  py::class_<Location>(m, "location", py::module_local())
      .def("__str__", [](Location &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<Value>(m, "value", py::module_local())
      .def("set_attr",
           [](Value &self, std::string &name, Attribute &attr) -> void {
             if (Operation *definingOp = self.getDefiningOp())
               definingOp->setAttr(name, attr);
             else {
               auto arg = mlir::cast<BlockArgument>(self);
               int id = arg.getArgNumber();
               std::string attrName = name + "_arg" + std::to_string(id);
               Block *owner = arg.getOwner();
               if (owner->isEntryBlock() &&
                   !isa<FuncOp>(owner->getParentOp())) {
                 owner->getParentOp()->setAttr(attrName, attr);
               }
             }
           })
      .def("get_context", &Value::getContext)
      .def("replace_all_uses_with",
           [](Value &self, Value &newValue) {
             self.replaceAllUsesWith(newValue);
           })
      .def("get_type", &Value::getType)
      .def("id", [](Value &self) {
        // The Value is identified by and compared with
        // other Values via the underlying ValueImpl
        return (uint64_t)self.getImpl();
      });

  py::class_<OpResult, Value>(m, "op_result", py::module_local());

  py::class_<BlockArgument, Value>(m, "block_argument", py::module_local());

  py::class_<Region>(m, "region", py::module_local())
      .def("get_parent_region", &Region::getParentRegion, ret::reference)
      .def("size", [](Region &self) { return self.getBlocks().size(); })
      .def("empty", &Region::empty)
      .def("id", [](Region &self) { return (uint64_t)&self; })
      .def("push_back",
           [](Region &self, Block *block) { self.push_back(block); })
      .def("push_front",
           [](Region &self, Block *block) { self.push_front(block); });

  py::class_<Block>(m, "block", py::module_local())
      .def("arg",
           [](Block &self, int index) -> BlockArgument {
             if (index >= self.getNumArguments())
               throw pybind11::index_error("Block argument index out of range");
             return self.getArgument(index);
           })
      .def("add_argument",
           [](Block &self, Type ty) {
             auto loc = UnknownLoc::get(ty.getContext());
             self.addArgument(ty, loc);
           })
      .def("get_num_arguments", &Block::getNumArguments)
      .def("get_argument", &Block::getArgument)
      .def("dump", &Block::dump)
      .def("move_before",
           [](Block &self, Block &dst) { self.moveBefore(&dst); })
      .def("insert_before", &Block::insertBefore)
      .def("get_parent", &Block::getParent, ret::reference)
      .def("merge_block_before",
           [](Block &self, Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](Block &self, Value &v, Value &newVal) {
             v.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
               Operation *user = operand.getOwner();
               Block *currentBlock = user->getBlock();
               while (currentBlock) {
                 if (currentBlock == &self)
                   return true;
                 // Move up one level
                 currentBlock =
                     currentBlock->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("has_terminator",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::IsTerminator>();
           })
      .def("has_return",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::ReturnLike>();
           })
      .def("erase", [](Block &self) { self.erase(); })
      .def("id", [](Block &self) { return (uint64_t)&self; });

  py::class_<Attribute>(m, "attribute", py::module_local());
  py::class_<IntegerAttr, Attribute>(m, "integer_attr", py::module_local());
  py::class_<BoolAttr, Attribute>(m, "bool_attr", py::module_local());
  py::class_<UnitAttr, Attribute>(m, "unit_attr", py::module_local());

  // Ops
  py::class_<OpState>(m, "OpState", py::module_local())
      .def("set_attr",
           [](OpState &self, std::string &name, Attribute &attr) -> void {
             self->setAttr(name, attr);
           })
      .def("get_num_results",
           [](OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](OpState &self, unsigned idx) -> Value {
             if (idx >= self->getNumResults())
               throw pybind11::index_error("Op result index out of range");
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](OpState &self, unsigned idx) -> Region & {
            if (idx >= self->getNumRegions())
              throw pybind11::index_error("Op region index out of range");
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](scf::ForOp &self, unsigned idx) -> Block * {
            if (idx >= self->getNumRegions())
              throw pybind11::index_error("Op region index out of range");
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](OpState &self) { self->dump(); })
      .def("__str__",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = getOpPrintingFlags();
             self->print(os, printingFlags);
             return str;
           })
      .def("str_nodebug",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self->print(os);
             return str;
           })
      .def("append_operand",
           [](OpState &self, Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify",
           [](OpState &self) -> bool {
             return succeeded(verify(self.getOperation()));
           })
      .def("get_operation", [](OpState &self) { return self.getOperation(); });

  // scf Ops
  py::class_<scf::ForOp, OpState>(m, "ForOp", py::module_local())
      .def("get_induction_var", &scf::ForOp::getInductionVar);

  py::class_<scf::IfOp, OpState>(m, "IfOp", py::module_local())
      .def("get_then_block", &scf::IfOp::thenBlock, ret::reference)
      .def("get_else_block", &scf::IfOp::elseBlock, ret::reference)
      .def("get_then_yield", &scf::IfOp::thenYield)
      .def("get_else_yield", &scf::IfOp::elseYield);
  py::class_<scf::YieldOp, OpState>(m, "YieldOp", py::module_local());
  py::class_<scf::WhileOp, OpState>(m, "WhileOp", py::module_local())
      .def("get_before", &scf::WhileOp::getBefore, ret::reference)
      .def("get_after", &scf::WhileOp::getAfter, ret::reference);
  py::class_<scf::ConditionOp, OpState>(m, "ConditionOp", py::module_local());

  py::class_<Operation, std::unique_ptr<Operation, py::nodelete>>(
      m, "operation", py::module_local())
      .def("get_name",
           [](Operation &self) {
             llvm::StringRef opName = self.getName().getStringRef();
             return opName.str();
           })
      .def("get_num_operands", &Operation::getNumOperands)
      .def("get_operand", &Operation::getOperand)
      .def("get_num_results", &Operation::getNumResults)
      .def("get_result", &Operation::getResult)
      .def("get_num_regions", &Operation::getNumRegions)
      .def("get_region", &Operation::getRegion, ret::reference)
      .def("get_block", &Operation::getBlock, ret::reference)
      .def("get_str_attr",
           [](Operation &self, const std::string &name) -> py::object {
             auto ret = self.getAttrOfType<StringAttr>(name);
             if (!ret)
               return py::none();
             return py::str(ret.getValue().str());
           })
      .def("get_bool_attr",
           [](Operation &self, const std::string &name) -> py::object {
             auto ret = self.getAttrOfType<BoolAttr>(name);
             if (!ret)
               return py::none();
             return py::bool_(ret.getValue());
           })
      .def("get_flat_symbol_ref_attr",
           [](Operation &self, const std::string &name) -> py::object {
             auto ret = self.getAttrOfType<FlatSymbolRefAttr>(name);
             if (!ret)
               return py::none();
             return py::str(ret.getValue().str());
           });

  // dynamic_attr is used to transfer ownership of the MLIR context to the
  // module
  py::class_<ModuleOp, OpState>(m, "module", py::module_local(),
                                py::dynamic_attr())
      .def("dump", &ModuleOp::dump)
      .def("str",
           [](ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = getOpPrintingFlags();
             self.print(os, printingFlags);
             return str;
           })
      .def("push_back",
           [](ModuleOp &self, FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("get_entry_func_name",
           [](ModuleOp &self) -> std::string {
             for (auto &op : self.getOps()) {
               if (auto func = dyn_cast<FuncOp>(op)) {
                 if (triton::isKernel(func))
                   return func.getName().str();
               }
             }
             return "";
           })
      .def("has_function",
           [](ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](ModuleOp &self, std::string &funcName) -> FuncOp {
             return self.lookupSymbol<FuncOp>(funcName);
           })
      /*
       * def ty_to_cpp(ty) is the consumer of this function.
       * If the type is a ptr it expects ty[0] == '*', else the type itself.
       */

      .def("get_function_signature",
           [](ModuleOp &self, FuncOp &func) -> std::vector<std::string> {
             std::vector<std::string> strVec;

             auto type = func.getFunctionType();
             unsigned numArgs = type.getNumInputs();
             for (unsigned i = 0; i != numArgs; ++i) {
               std::string tempType;
               llvm::raw_string_ostream os(tempType);

               auto ty = type.getInput(i);
               if (auto attributes = func.getCallableArgAttrs()) {
                 Attribute attr = attributes[i];
                 // Check for tt.nv_tma_desc = 1
                 if (auto dAttr = dyn_cast<DictionaryAttr>(attr)) {
                   if (dAttr.contains("tt.nv_tma_desc")) {
                     strVec.push_back("nvTmaDesc");
                     continue;
                   }
                 }
               }
               if (auto ptrType = dyn_cast<PointerType>(ty)) {
                 auto pType = ptrType.getPointeeType();
                 os << "*";
                 pType.print(os);
               } else {
                 ty.print(os);
               }
               strVec.push_back(tempType);
             }
             return strVec;
           })
      .def("get_int_attr",
           [](ModuleOp &self, std::string name) -> py::object {
             auto ret = self->getAttrOfType<IntegerAttr>(name);
             if (!ret)
               return py::none();
             return py::int_(ret.getInt());
           })
      .def("get_tensordesc_metadata", getTensorDescMetadata)
      .def("create_location_snapshot",
           [](ModuleOp &self, const std::string &fileName) -> void {
             auto printingFlags = getOpPrintingFlags();
             if (failed(generateLocationsFromIR(fileName, self, printingFlags)))
               throw std::runtime_error("Failed to create location snapshot");
           })
      .def("walk",
           [](ModuleOp &self, const std::function<void(Operation *)> &fn) {
             self.walk(fn);
           });

  m.def("make_attr", [](const std::vector<int> &values, MLIRContext &context) {
    return mlir::cast<Attribute>(DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(values.size())},
                              IntegerType::get(&context, 32)),
        values));
  });

  m.def(
      "parse_mlir_module",
      [](const std::string &inputFilename, MLIRContext &context) {
        // parse module
        OwningOpRef<ModuleOp> module =
            parseSourceFile<ModuleOp>(inputFilename, &context);
        if (!module)
          throw std::runtime_error("Parse MLIR file failed.");
        return module->clone();
      },
      ret::take_ownership);

  py::class_<FuncOp, OpState>(m, "function", py::module_local())
      // .def_property_readonly("attrs", &ir::function::attrs)
      // .def("add_attr", &ir::function::add_attr);
      .def("args",
           [](FuncOp &self, unsigned idx) -> BlockArgument {
             if (idx >= self.getNumArguments())
               throw pybind11::index_error(
                   "Function argument index out of range");
             return self.getArgument(idx);
           })
      .def("get_num_args", &FuncOp::getNumArguments)
      .def(
          "add_entry_block",
          [](FuncOp &self) -> Block * { return self.addEntryBlock(); },
          ret::reference)
      .def(
          "set_arg_attr",
          [](FuncOp &self, int arg_no, const std::string &name, int val) {
            if (arg_no >= self.getNumArguments())
              throw pybind11::index_error(
                  "Function argument index out of range");
            // set arg attributes "name" to value "val"
            auto attrTy = IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      //  .def("has_attr", &::FuncOp::hasAttr)
      .def("finalize", [](FuncOp &self) -> void {})
      .def_property_readonly("type", &FuncOp::getFunctionType)
      .def("reset_type", &FuncOp::setType);

  py::class_<OpBuilder::InsertPoint>(m, "InsertPoint", py::module_local());

  py::class_<TritonOpBuilder>(m, "builder", py::module_local(),
                              py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      // getters
      .def("create_module",
           [](TritonOpBuilder &self) -> ModuleOp {
             return self.create<ModuleOp>();
           })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](TritonOpBuilder &self, Block &block) -> void {
             self.setInsertionPointToStart(block);
           })
      .def("set_insertion_point_to_end",
           [](TritonOpBuilder &self, Block &block) {
             self.setInsertionPointToEnd(block);
           })
      .def("set_insertion_point_after",
           [](TritonOpBuilder &self, Operation &op) {
             self.setInsertionPointAfter(op);
           })
      .def(
          "get_insertion_block",
          [](TritonOpBuilder &self) -> Block * {
            return self.getBuilder().getInsertionBlock();
          },
          ret::reference)
      .def("get_insertion_point",
           [](TritonOpBuilder &self) {
             return self.getBuilder().saveInsertionPoint();
           })
      .def("restore_insertion_point",
           [](TritonOpBuilder &self, OpBuilder::InsertPoint pt) {
             self.restoreInsertionPoint(pt);
           })
      .def("set_async_task_ids",
           [](TritonOpBuilder &self, std::vector<int> v) {
             self.setAsyncTaskIds(v);
           })
      .def("unset_async_task_ids",
           [](TritonOpBuilder &self) { self.unsetAsyncTaskIds(); })
      // Attr
      .def(
          "get_unit_attr",
          [](TritonOpBuilder &self) { return self.getBuilder().getUnitAttr(); })
      .def("get_bool_attr",
           [](TritonOpBuilder &self, bool value) {
             return self.getBuilder().getBoolAttr(value);
           })
      .def("get_int32_attr",
           [](TritonOpBuilder &self, int32_t value) {
             return self.getBuilder().getI32IntegerAttr(value);
           })
      .def("get_string_attr",
           [](TritonOpBuilder &self, std::string value) -> Attribute {
             return self.getBuilder().getStringAttr(value);
           })
      // Use arith.ConstantOp to create constants
      // Constants
      .def("get_int1",
           [](TritonOpBuilder &self, bool v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI1Type()));
           })
      .def("get_int8",
           [](TritonOpBuilder &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI8Type()));
           })
      .def("get_int16",
           [](TritonOpBuilder &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI16Type()));
           })
      .def("get_int32",
           [](TritonOpBuilder &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI32Type()));
           })
      .def("get_int64",
           [](TritonOpBuilder &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI64Type()));
           })
      .def("get_uint8",
           [](TritonOpBuilder &self, uint64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI8Type()));
           })
      .def("get_uint16",
           [](TritonOpBuilder &self, uint64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI16Type()));
           })
      .def("get_uint32",
           [](TritonOpBuilder &self, uint64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI32Type()));
           })
      .def("get_uint64",
           [](TritonOpBuilder &self, uint64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(
                 v, self.getBuilder().getI64Type()));
           })
      .def("get_bf16",
           [](TritonOpBuilder &self, float v) -> Value {
             auto type = self.getBuilder().getBF16Type();
             return self.create<arith::ConstantFloatOp>(
                 APFloat(type.getFloatSemantics(), std::to_string(v)), type);
           })
      .def("get_fp16",
           [](TritonOpBuilder &self, float v) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF16FloatAttr(v));
           })
      .def("get_fp32",
           [](TritonOpBuilder &self, float v) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF32FloatAttr(v));
           })
      .def("get_fp64",
           [](TritonOpBuilder &self, double v) -> Value {
             return self.create<arith::ConstantOp>(
                 self.getBuilder().getF64FloatAttr(v));
           })
      .def("get_null_value",
           [](TritonOpBuilder &self, Type type) -> Value {
             if (auto floatTy = dyn_cast<FloatType>(type))
               return self.create<arith::ConstantFloatOp>(
                   APFloat(floatTy.getFloatSemantics(), 0), floatTy);
             else if (auto intTy = dyn_cast<IntegerType>(type))
               return self.create<arith::ConstantIntOp>(0, intTy);
             else
               throw std::runtime_error("Not implemented");
           })
      .def("get_all_ones_value",
           [](TritonOpBuilder &self, Type type) -> Value {
             uint64_t val = 0xFFFFFFFFFFFFFFFF;
             if (auto intTy = dyn_cast<IntegerType>(type))
               return self.create<arith::ConstantIntOp>(val, intTy);
             else
               throw std::runtime_error("Not implemented");
           })

      // Types
      .def("get_void_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getNoneType();
           })
      .def("get_int1_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI1Type();
           }) // or ret::copy?
      .def("get_int8_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI8Type();
           })
      .def("get_int16_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<IntegerType>(16);
           })
      .def("get_int32_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI32Type();
           })
      .def("get_int64_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI64Type();
           })
      .def("get_fp8e4nv_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<Float8E4M3FNType>();
           })
      .def("get_fp8e4b8_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<Float8E4M3FNUZType>();
           })
      .def("get_fp8e4b15_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI8Type();
           })
      .def("get_fp8e5_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<Float8E5M2Type>();
           })
      .def("get_fp8e5b16_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getType<Float8E5M2FNUZType>();
           })
      .def("get_half_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF16Type();
           })
      .def("get_bf16_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getBF16Type();
           })
      .def("get_float_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF32Type();
           })
      .def("get_double_ty",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF64Type();
           })
      .def("get_ptr_ty",
           [](TritonOpBuilder &self, Type &type, int addrSpace) -> Type {
             return PointerType::get(type, addrSpace);
           })
      .def("get_block_ty",
           [](TritonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape) -> Type {
             return RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty",
           [](TritonOpBuilder &self, std::vector<Type> inTypes,
              std::vector<Type> outTypes) -> Type {
             return self.getBuilder().getFunctionType(inTypes, outTypes);
           })
      // locs
      .def("set_loc",
           [](TritonOpBuilder &self, Location loc) { self.setLastLoc(loc); })
      .def("set_loc",
           [](TritonOpBuilder &self, const std::string &fileName, int line,
              int column) { self.setLastLoc(fileName, line, column); })
      .def("get_loc",
           [](TritonOpBuilder &self) -> Location { return self.getLastLoc(); })

      // Ops
      .def("get_or_insert_function",
           [](TritonOpBuilder &self, ModuleOp &module, std::string &funcName,
              Type &funcType, std::string &visibility,
              bool noinline) -> FuncOp {
             if (Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<FuncOp>(funcOperation);
             if (auto funcTy = dyn_cast<FunctionType>(funcType)) {
               llvm::SmallVector<NamedAttribute> attrs = {
                   NamedAttribute(
                       self.getBuilder().getStringAttr("sym_visibility"),
                       self.getBuilder().getStringAttr(visibility)),
                   NamedAttribute(self.getBuilder().getStringAttr("noinline"),
                                  self.getBuilder().getBoolAttr(noinline))};
               return self.create<FuncOp>(funcName, funcTy, attrs);
             }
             throw std::invalid_argument("invalid function type");
           })
      .def(
          "create_block",
          [](TritonOpBuilder &self) -> Block * {
            Region *parent = self.getBuilder().getBlock()->getParent();
            return self.getBuilder().createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](TritonOpBuilder &self, Region &parent,
             std::vector<Type> &argTypes) -> Block * {
            // TODO: update arg loc
            auto loc = self.getBuilder().getUnknownLoc();
            llvm::SmallVector<Location, 8> argLocs(argTypes.size(), loc);
            return self.getBuilder().createBlock(&parent, {}, argTypes,
                                                 argLocs);
          },
          ret::reference)
      .def(
          "new_block",
          [](TritonOpBuilder &self) -> Block * { return new Block(); },
          ret::reference)
      // Function
      .def("ret",
           [](TritonOpBuilder &self, std::vector<Value> &vals) -> OpState {
             return self.create<ReturnOp>(vals);
           })
      .def("call",
           [](TritonOpBuilder &self, FuncOp &func, std::vector<Value> &args)
               -> OpState { return self.create<CallOp>(func, args); })
      // Unstructured control flow
      .def("create_cond_branch",
           [](TritonOpBuilder &self, Value condition, Block *trueDest,
              Block *falseDest) -> OpState {
             return self.create<cf::CondBranchOp>(condition, trueDest,
                                                  falseDest);
           })
      .def("create_branch",
           [](TritonOpBuilder &self, Block *dest, std::vector<Value> &args)
               -> OpState { return self.create<cf::BranchOp>(dest, args); })
      // Structured control flow
      .def("create_for_op",
           [](TritonOpBuilder &self, Value &lb, Value &ub, Value &step,
              std::vector<Value> &initArgs) -> scf::ForOp {
             return self.create<scf::ForOp>(lb, ub, step, initArgs);
           })
      .def("create_if_op",
           [](TritonOpBuilder &self, std::vector<Type> &retTypes,
              Value &condition, bool withElse) -> scf::IfOp {
             return self.create<scf::IfOp>(retTypes, condition, withElse);
           })
      .def("create_yield_op",
           [](TritonOpBuilder &self, std::vector<Value> &yields)
               -> scf::YieldOp { return self.create<scf::YieldOp>(yields); })
      .def("create_while_op",
           [](TritonOpBuilder &self, std::vector<Type> &retTypes,
              std::vector<Value> &initArgs) -> scf::WhileOp {
             return self.create<scf::WhileOp>(retTypes, initArgs);
           })
      .def("create_condition_op",
           [](TritonOpBuilder &self, Value &cond,
              std::vector<Value> &args) -> scf::ConditionOp {
             return self.create<scf::ConditionOp>(cond, args);
           })

      // miscellaneous
      .def("create_make_range",
           [](TritonOpBuilder &self, Type retTy, int start, int end) -> Value {
             return self.create<MakeRangeOp>(retTy, start, end);
           })

      // Cast instructions
      // Conversions for custom FP types (FP8 and non-standard rounding modes)
      .def("create_fp_to_fp",
           [](TritonOpBuilder &self, Value &src, Type &dstType,
              std::optional<RoundingMode> roundingMode) -> Value {
             if (roundingMode.has_value())
               return self.create<FpToFpOp>(
                   dstType, src,
                   RoundingModeAttr::get(self.getBuilder().getContext(),
                                         roundingMode.value()));
             else
               return self.create<FpToFpOp>(dstType, src);
           })
      // Conversions for standard LLVM builtin types
      .def("create_bitcast",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<BitcastOp>(dstType, src);
           })
      .def("create_si_to_fp",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::SIToFPOp>(dstType, src);
           })
      .def("create_ui_to_fp",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::UIToFPOp>(dstType, src);
           })
      .def("create_fp_to_si",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::FPToSIOp>(dstType, src);
           })
      .def("create_fp_to_ui",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::FPToUIOp>(dstType, src);
           })
      .def("create_fp_ext",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::ExtFOp>(dstType, src);
           })
      .def("create_fp_trunc",
           [](TritonOpBuilder &self, Value &src, Type &dstType) -> Value {
             return self.create<arith::TruncFOp>(dstType, src);
           })
      .def("create_int_cast",
           [](TritonOpBuilder &self, Value &src, Type &dstType,
              bool isSigned) -> Value {
             // get element type if necessary
             Type srcType = src.getType();
             auto srcTensorType = dyn_cast<RankedTensorType>(srcType);
             auto dstTensorType = dyn_cast<RankedTensorType>(dstType);
             Type srcEltType = srcType;
             Type dstEltType = dstType;
             if (dstTensorType && srcTensorType) {
               dstEltType = dstTensorType.getElementType();
               srcEltType = srcTensorType.getElementType();
             }
             unsigned srcWidth = srcEltType.getIntOrFloatBitWidth();
             unsigned dstWidth = dstEltType.getIntOrFloatBitWidth();
             if (srcWidth == dstWidth)
               return self.create<arith::BitcastOp>(dstType, src);
             else if (srcWidth > dstWidth)
               return self.create<arith::TruncIOp>(dstType, src);
             else if (isSigned)
               return self.create<arith::ExtSIOp>(dstType, src);
             else
               return self.create<arith::ExtUIOp>(dstType, src);
           })
      .def("create_fmul",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MulFOp>(lhs, rhs);
           })
      .def("create_fdiv",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivFOp>(lhs, rhs);
           })
      .def("create_frem",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemFOp>(lhs, rhs);
           })
      .def("create_fadd",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AddFOp>(lhs, rhs);
           })
      .def("create_fsub",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::SubFOp>(lhs, rhs);
           })
      .def("create_mul",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::MulIOp>(lhs, rhs);
           })
      .def("create_umulhi",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<triton::MulhiUIOp>(lhs, rhs);
           })
      .def("create_sdiv",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivSIOp>(lhs, rhs);
           })
      .def("create_udiv",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::DivUIOp>(lhs, rhs);
           })
      .def("create_srem",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemSIOp>(lhs, rhs);
           })
      .def("create_urem",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::RemUIOp>(lhs, rhs);
           })
      .def("create_add",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AddIOp>(lhs, rhs);
           })
      .def("create_sub",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::SubIOp>(lhs, rhs));
           })
      .def("create_fma",
           [](TritonOpBuilder &self, Value &a, Value &b, Value &c) -> Value {
             return Value(self.create<math::FmaOp>(a, b, c));
           })
      .def("create_shl",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::ShLIOp>(lhs, rhs));
           })
      .def("create_lshr",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::ShRUIOp>(lhs, rhs));
           })
      .def("create_ashr",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::ShRSIOp>(lhs, rhs));
           })
      .def("create_minsi",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MinSIOp>(lhs, rhs));
           })
      .def("create_minui",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MinUIOp>(lhs, rhs));
           })
      // minimumf follows the torch.minimum convention and returns NaN if either
      // operand is NaN
      .def("create_minimumf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MinimumFOp>(lhs, rhs));
           })
      // minnumf follows the torch.fmin convention and returns the non-NaN
      // operand
      .def("create_minnumf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MinNumFOp>(lhs, rhs));
           })
      .def("create_maxsi",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MaxSIOp>(lhs, rhs));
           })
      .def("create_maxui",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MaxUIOp>(lhs, rhs));
           })
      // maximumf follows the torch.maximum convention and returns NaN if either
      // operand is NaN
      .def("create_maximumf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MaximumFOp>(lhs, rhs));
           })
      // maxnumf follows the torch.fmax convention and returns the non-NaN
      // operand
      .def("create_maxnumf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<arith::MaxNumFOp>(lhs, rhs));
           })
      .def("create_clampf",
           [](TritonOpBuilder &self, Value &input, Value &min, Value &max,
              PropagateNan propagateNan) -> Value {
             return Value(self.create<ClampFOp>(input, min, max, propagateNan));
           })
      .def("create_precise_sqrt",
           [](TritonOpBuilder &self, Value &input) -> Value {
             return Value(self.create<PreciseSqrtOp>(input));
           })
      .def("create_precise_divf",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return Value(self.create<PreciseDivFOp>(lhs, rhs));
           })
      // AddPtr (similar to GEP)
      .def("create_addptr",
           [](TritonOpBuilder &self, Value &ptr, Value &offset) -> Value {
             return self.create<AddPtrOp>(ptr.getType(), ptr, offset);
           })
      // Comparison (int)
      .def("create_icmpSLE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sle, lhs,
                                               rhs);
           })
      .def("create_icmpSLT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::slt, lhs,
                                               rhs);
           })
      .def("create_icmpSGE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sge, lhs,
                                               rhs);
           })
      .def("create_icmpSGT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::sgt, lhs,
                                               rhs);
           })
      .def("create_icmpULE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ule, lhs,
                                               rhs);
           })
      .def("create_icmpULT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ult, lhs,
                                               rhs);
           })
      .def("create_icmpUGE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::uge, lhs,
                                               rhs);
           })
      .def("create_icmpUGT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ugt, lhs,
                                               rhs);
           })
      .def("create_icmpEQ",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs,
                                               rhs);
           })
      .def("create_icmpNE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpIOp>(arith::CmpIPredicate::ne, lhs,
                                               rhs);
           })
      // Comparison (float)
      .def("create_fcmpOLT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, lhs,
                                               rhs);
           })
      .def("create_fcmpOGT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, lhs,
                                               rhs);
           })
      .def("create_fcmpOLE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OLE, lhs,
                                               rhs);
           })
      .def("create_fcmpOGE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OGE, lhs,
                                               rhs);
           })
      .def("create_fcmpOEQ",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, lhs,
                                               rhs);
           })
      .def("create_fcmpONE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ONE, lhs,
                                               rhs);
           })
      .def("create_fcmpULT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ULT, lhs,
                                               rhs);
           })
      .def("create_fcmpUGT",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UGT, lhs,
                                               rhs);
           })
      .def("create_fcmpULE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::ULE, lhs,
                                               rhs);
           })
      .def("create_fcmpUGE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UGE, lhs,
                                               rhs);
           })
      .def("create_fcmpUEQ",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UEQ, lhs,
                                               rhs);
           })
      .def("create_fcmpUNE",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::CmpFOp>(arith::CmpFPredicate::UNE, lhs,
                                               rhs);
           })
      // // Logical
      .def("create_and",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::AndIOp>(lhs, rhs);
           })
      .def("create_xor",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::XOrIOp>(lhs, rhs);
           })
      .def("create_or",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             return self.create<arith::OrIOp>(lhs, rhs);
           })
      // Input/Output
      .def("create_load",
           [](TritonOpBuilder &self, Value &ptrs, CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy, bool isVolatile) -> Value {
             return self.create<LoadOp>(ptrs, cacheModifier, evictionPolicy,
                                        isVolatile);
           })
      .def("create_store",
           [](TritonOpBuilder &self, Value &ptrs, Value &value,
              CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy) -> void {
             self.create<StoreOp>(ptrs, value, cacheModifier, evictionPolicy);
           })
      .def("create_tensor_pointer_load",
           [](TritonOpBuilder &self, Value &ptr,
              std::vector<int32_t> &boundaryCheck,
              std::optional<PaddingOption> paddingOption,
              CacheModifier cacheModifier, EvictionPolicy evictionPolicy,
              bool isVolatile) -> Value {
             return self.create<LoadOp>(ptr, boundaryCheck, paddingOption,
                                        cacheModifier, evictionPolicy,
                                        isVolatile);
           })
      .def("create_tensor_pointer_store",
           [](TritonOpBuilder &self, Value &ptr, Value &val,
              std::vector<int32_t> &boundaryCheck, CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy) -> void {
             self.create<StoreOp>(ptr, val, boundaryCheck, cacheModifier,
                                  evictionPolicy);
           })
      .def("create_masked_load",
           [](TritonOpBuilder &self, Value &ptrs, Value &mask,
              std::optional<Value> &other, CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy, bool isVolatile) -> Value {
             return self.create<LoadOp>(ptrs, mask, other.value_or(Value()),
                                        cacheModifier, evictionPolicy,
                                        isVolatile);
           })
      .def("create_masked_store",
           [](TritonOpBuilder &self, Value &ptrs, Value &val, Value &mask,
              CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy) -> void {
             self.create<StoreOp>(ptrs, val, mask, cacheModifier,
                                  evictionPolicy);
           })
      .def("create_tensor_descriptor_type",
           [](TritonOpBuilder &self, Type blockTy, bool isSigned) -> Type {
             auto ctx = self.getContext();
             return triton::TensorDescType::get(
                 ctx, cast<RankedTensorType>(blockTy), isSigned);
           })
      .def("create_descriptor_load",
           [](TritonOpBuilder &self, Value desc, std::vector<Value> &indices,
              CacheModifier cacheModifier,
              EvictionPolicy evictionPolicy) -> Value {
             auto descTy = cast<triton::TensorDescType>(desc.getType());
             auto resTy = descTy.getSignlessBlockType();
             return self.create<DescriptorLoadOp>(
                 resTy, desc, indices, cacheModifier, evictionPolicy);
           })
      .def("create_descriptor_gather",
           [](TritonOpBuilder &self, Value desc, Value x_indices, Value y_index,
              Type type) -> Value {
             return self.create<DescriptorGatherOp>(type, desc, x_indices,
                                                    y_index);
           })
      .def("create_descriptor_store",
           [](TritonOpBuilder &self, Value desc, Value value,
              std::vector<Value> &indices) -> void {
             self.create<DescriptorStoreOp>(desc, value, indices);
           })
      .def("create_descriptor_reduce",
           [](TritonOpBuilder &self, DescriptorReduceKind kind, Value desc,
              Value value, std::vector<Value> &indices) -> void {
             self.create<DescriptorReduceOp>(kind, desc, value, indices);
           })
      .def("create_descriptor_scatter",
           [](TritonOpBuilder &self, Value desc, Value value, Value x_indices,
              Value y_index) -> void {
             self.create<DescriptorScatterOp>(desc, x_indices, y_index, value);
           })
      .def("create_reshape",
           [](TritonOpBuilder &self, Value &arg, std::vector<int64_t> &shape,
              bool allowReorder) -> Value {
             return self.create<ReshapeOp>(shape, arg, allowReorder);
           })
      .def("create_expand_dims",
           [](TritonOpBuilder &self, Value &arg, int axis) -> Value {
             return self.create<ExpandDimsOp>(arg, axis);
           })
      .def("create_cat",
           [](TritonOpBuilder &self, Value &lhs, Value &rhs) -> Value {
             auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
             auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
             if (!(lhsType.getShape().size() == 1 &&
                   rhsType.getShape().size() == 1))
               throw std::invalid_argument(
                   "shape not supported by cat. Expecting rank-1 inputs");
             std::vector<int64_t> shape{lhsType.getShape()[0] +
                                        rhsType.getShape()[0]};
             return self.create<CatOp>(
                 RankedTensorType::get(shape, lhsType.getElementType()), lhs,
                 rhs);
           })
      .def("create_join",
           [](TritonOpBuilder &self, Value &a, Value &b) -> Value {
             return self.create<JoinOp>(a, b);
           })
      .def("create_split",
           [](TritonOpBuilder &self, Value &a) -> std::vector<Value> {
             auto op = self.create<SplitOp>(a);
             return std::vector<Value>(op->result_begin(), op->result_end());
           })
      // Implements tl.trans and tl.permute.
      .def("create_trans",
           [](TritonOpBuilder &self, Value &arg, std::vector<int> &order)
               -> Value { return self.create<TransOp>(arg, order); })
      .def("create_broadcast",
           [](TritonOpBuilder &self, Value &arg,
              std::vector<int64_t> &shape) -> Value {
             if (auto argType = dyn_cast<RankedTensorType>(arg.getType()))
               return self.createOrFold<BroadcastOp>(
                   RankedTensorType::get(shape, argType.getElementType()), arg);
             throw std::invalid_argument(
                 "arg is not of RankedTensorType, use create_splat");
           })
      .def("create_splat",
           [](TritonOpBuilder &self, Type &retTy, Value &arg) -> Value {
             return self.createOrFold<SplatOp>(retTy, arg);
           })
      // // atomic
      .def("create_atomic_cas",
           [](TritonOpBuilder &self, Value &ptr, Value &cmp, Value &val,
              MemSemantic sem, MemSyncScope scope) -> Value {
             Type dstType;
             if (auto srcTensorType =
                     dyn_cast<RankedTensorType>(ptr.getType())) {
               Type dstElemType =
                   cast<PointerType>(srcTensorType.getElementType())
                       .getPointeeType();
               dstType =
                   RankedTensorType::get(srcTensorType.getShape(), dstElemType);
             } else {
               auto ptrType = cast<PointerType>(getElementTypeOrSelf(ptr));
               dstType = ptrType.getPointeeType();
             }
             return self.create<AtomicCASOp>(dstType, ptr, cmp, val, sem,
                                             scope);
           })
      .def("create_atomic_rmw",
           [](TritonOpBuilder &self, RMWOp rmwOp, Value &ptr, Value &val,
              Value &mask, MemSemantic sem, MemSyncScope scope) -> Value {
             Type dstType;
             if (auto srcTensorType =
                     dyn_cast<RankedTensorType>(ptr.getType())) {
               Type dstElemType =
                   cast<PointerType>(srcTensorType.getElementType())
                       .getPointeeType();
               dstType =
                   RankedTensorType::get(srcTensorType.getShape(), dstElemType);
             } else {
               auto ptrType = cast<PointerType>(getElementTypeOrSelf(ptr));
               dstType = ptrType.getPointeeType();
             }
             return self.create<AtomicRMWOp>(dstType, rmwOp, ptr, val, mask,
                                             sem, scope);
           })
      // External
      .def("create_extern_elementwise",
           [](TritonOpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<Value> &argList, Type retType, bool isPure) -> Value {
             return self.create<ExternElementwiseOp>(retType, argList, libName,
                                                     libPath, symbol, isPure);
           })
      // Built-in instruction
      .def("create_get_program_id",
           [](TritonOpBuilder &self, int axis) -> Value {
             if (axis < 0 || axis > 3)
               throw pybind11::index_error("program_id must be in [0,3]");
             return self.create<GetProgramIdOp>(axis);
           })
      .def("create_get_num_programs",
           [](TritonOpBuilder &self, int axis) -> Value {
             if (axis < 0 || axis > 3)
               throw pybind11::index_error("program_id must be in [0,3]");
             return self.create<GetNumProgramsOp>(axis);
           })
      .def("create_dot",
           [](TritonOpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, InputPrecision inputPrecision,
              int maxNumImpreciseAcc) -> mlir::Value {
             return self.create<DotOp>(c.getType(), a, b, c, inputPrecision,
                                       maxNumImpreciseAcc);
           })
      .def("create_dot_scaled",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              std::optional<mlir::Value> &lhs_scale,
              ScaleDotElemType lhs_format, mlir::Value &rhs,
              std::optional<mlir::Value> &rhs_scale,
              ScaleDotElemType rhs_format, bool fast_math, bool lhs_k_pack,
              bool rhs_k_pack, mlir::Value &c) -> mlir::Value {
             return self.create<DotScaledOp>(
                 c.getType(), lhs, rhs, c, lhs_scale.value_or(Value()),
                 rhs_scale.value_or(Value()), lhs_format, rhs_format, fast_math,
                 lhs_k_pack, rhs_k_pack);
           })
      .def("create_floor",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::FloorOp>(val);
           })
      .def("create_ceil",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::CeilOp>(val);
           })
      .def("create_exp",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::ExpOp>(val);
           })
      .def("create_exp2",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::Exp2Op>(val);
           })
      .def("create_cos",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::CosOp>(val);
           })
      .def("create_sin",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::SinOp>(val);
           })
      .def("create_log",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::LogOp>(val);
           })
      .def("create_log2",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::Log2Op>(val);
           })
      .def("create_erf",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::ErfOp>(val);
           })
      .def("create_sqrt",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::SqrtOp>(val);
           })
      .def("create_rsqrt",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::RsqrtOp>(val);
           })
      .def("create_fabs",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::AbsFOp>(val);
           })
      .def("create_iabs",
           [](TritonOpBuilder &self, Value &val) -> Value {
             return self.create<math::AbsIOp>(val);
           })
      .def("create_reduce",
           [](TritonOpBuilder &self, std::vector<Value> operands, int axis)
               -> OpState { return self.create<ReduceOp>(operands, axis); })
      .def("create_reduce_ret",
           [](TritonOpBuilder &self, py::args args) -> OpState {
             llvm::SmallVector<Value> return_values;
             for (const auto &arg : args) {
               return_values.push_back(py::cast<Value>(arg));
             }
             return self.create<ReduceReturnOp>(return_values);
           })
      .def("create_scan",
           [](TritonOpBuilder &self, std::vector<Value> operands, int axis,
              bool reverse) -> OpState {
             return self.create<ScanOp>(operands, axis, reverse);
           })
      .def("create_scan_ret",
           [](TritonOpBuilder &self, py::args args) -> OpState {
             llvm::SmallVector<Value> return_values;
             for (const auto &arg : args) {
               return_values.push_back(py::cast<Value>(arg));
             }
             return self.create<ScanReturnOp>(return_values);
           })
      .def("create_ptr_to_int",
           [](TritonOpBuilder &self, Value &val, Type &type) -> Value {
             return self.create<PtrToIntOp>(type, val);
           })
      .def("create_int_to_ptr",
           [](TritonOpBuilder &self, Value &val, Type &type) -> Value {
             return self.create<IntToPtrOp>(type, val);
           })
      .def("create_select",
           [](TritonOpBuilder &self, Value &condition, Value &trueValue,
              Value &falseValue) -> Value {
             return self.create<arith::SelectOp>(condition, trueValue,
                                                 falseValue);
           })
      .def("create_inline_asm",
           [](TritonOpBuilder &self, const std::string &inlineAsm,
              const std::string &constraints, const std::vector<Value> &values,
              const std::vector<Type> &types, bool isPure,
              int pack) -> OpState {
             return self.create<ElementwiseInlineAsmOp>(
                 types, inlineAsm, constraints, isPure, pack, values);
           })
      .def("create_print",
           [](TritonOpBuilder &self, const std::string &prefix, bool hex,
              const std::vector<Value> &values,
              const std::vector<int32_t> &isSigned) -> void {
             auto prefixAttr = StringAttr::get(self.getBuilder().getContext(),
                                               llvm::StringRef(prefix));
             self.create<PrintOp>(prefixAttr, hex, values, isSigned);
           })
      .def("create_assert",
           [](TritonOpBuilder &self, Value &condition,
              const std::string &message) -> void {
             auto messageAttr = StringAttr::get(self.getBuilder().getContext(),
                                                llvm::StringRef(message));
             self.create<AssertOp>(condition, messageAttr);
           })
      .def("create_assume",
           [](TritonOpBuilder &self, Value &condition) {
             self.create<LLVM::AssumeOp>(condition);
           })
      .def("create_poison",
           [](TritonOpBuilder &self, Type &type) -> Value {
             return self.create<ub::PoisonOp>(type);
           })
      .def("create_histogram",
           [](TritonOpBuilder &self, Value operand, int numBins,
              std::optional<Value> mask) -> Value {
             if (!mask) {
               return self.create<HistogramOp>(
                   RankedTensorType::get(
                       {static_cast<int64_t>(numBins)},
                       IntegerType::get(operand.getContext(), 32)),
                   operand);
             } else {
               return self.create<HistogramOp>(
                   RankedTensorType::get(
                       {static_cast<int64_t>(numBins)},
                       IntegerType::get(operand.getContext(), 32)),
                   operand, *mask);
             }
           })
      .def("create_gather",
           [](TritonOpBuilder &self, Value src, Value indices, int axis)
               -> Value { return self.create<GatherOp>(src, indices, axis); })
      // Force GPU barrier
      .def("create_barrier",
           [](TritonOpBuilder &self) { self.create<mlir::gpu::BarrierOp>(); })
      // Make a block pointer (tensor pointer in Triton IR)
      .def("create_make_block_ptr",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &shape,
              std::vector<Value> &strides, std::vector<Value> &offsets,
              std::vector<int32_t> &tensorShape,
              std::vector<int32_t> &order) -> Value {
             return self.create<MakeTensorPtrOp>(base, shape, strides, offsets,
                                                 tensorShape, order);
           })
      // Advance a block pointer
      .def("create_advance",
           [](TritonOpBuilder &self, Value &ptr,
              std::vector<Value> &offsets) -> Value {
             return self.create<AdvanceOp>(ptr.getType(), ptr, offsets);
           })
      // Make a tensor descriptor
      .def("create_make_tensor_descriptor",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &shape,
              std::vector<Value> &strides, std::vector<int32_t> &tensorShape,
              bool isSignedInteger) -> Value {
             return self.create<MakeTensorDescOp>(base, shape, strides,
                                                  tensorShape, isSignedInteger);
           })
      // Proton Ops
      .def("create_proton_record",
           [](TritonOpBuilder &self, bool isStart, int32_t regionId) -> void {
             self.create<mlir::triton::proton::RecordOp>(isStart, regionId);
           });

  py::class_<PassManager>(m, "pass_manager", py::module_local())
      .def(py::init<MLIRContext *>())
      .def("enable_debug",
           [](PassManager &self) -> bool {
             auto *context = self.getContext();
             bool haveDump = ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
             std::string funcToDump;
             if (!haveDump) {
               funcToDump = triton::tools::getStrEnv("MLIR_ENABLE_DUMP");
               bool isEnvValueBool =
                   triton::tools::isEnvValueBool(funcToDump).has_value();
               if (!funcToDump.empty() && !isEnvValueBool)
                 haveDump = true;
             }
             if (haveDump) {
               context->disableMultithreading();
               auto printingFlags = getOpPrintingFlags();
               auto printAlways = [funcToDump](Pass *, Operation *op) -> bool {
                 if (funcToDump.empty())
                   return true;
                 if (auto mod = dyn_cast<mlir::ModuleOp>(op)) {
                   return mod.lookupSymbol(funcToDump);
                 }
                 if (auto func = dyn_cast<triton::FuncOp>(op)) {
                   return SymbolTable::getSymbolName(func).getValue() ==
                          funcToDump;
                 }

                 return false;
               };
               self.enableIRPrinting(
                   /*shouldPrintBeforePass=*/printAlways,
                   /*shouldPrintAfterPass=*/printAlways,
                   /*printModuleScope=*/true,
                   /*printAfterOnlyOnChange=*/false,
                   /*printAfterOnlyOnFailure*/ true, mlir_dumps_or_dbgs(),
                   printingFlags);
             }
             return haveDump;
           })
      .def("get_pipeline_str",
           [](PassManager &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.printAsTextualPipeline(os);
             return str;
           })
      .def("run", [](PassManager &self, ModuleOp &mod) {
        // TODO: maybe dump module to file and print error for better
        // diagnostics

        auto *context = mod.getContext();
        if (::triton::tools::getBoolEnv("MLIR_DISABLE_MULTITHREADING"))
          context->disableMultithreading();

        auto reproducerPath =
            triton::tools::getStrEnv("TRITON_REPRODUCER_PATH");
        if (!reproducerPath.empty()) {
          auto anchorName = self.getOpAnchorName();
          auto passes = self.getPasses();
          Operation *op = mod.getOperation();
          // Save a reproducer for the current pass manager invocation
          // immediately.
          makeReproducer(anchorName, passes, op, reproducerPath);
          // But if the pass manager crashes, attempt to generate a local
          // reproducer instead.
          context->disableMultithreading();
          self.enableCrashReproducerGeneration(reproducerPath,
                                               /*genLocalReproducer=*/true);
        } else {
          self.enableCrashReproducerGeneration(makeConsoleReproducer());
        }

        if (triton::tools::getBoolEnv("TRITON_ENABLE_LLVM_DEBUG")) {
          ::llvm::DebugFlag = true;
        }

        if (auto debugOnly = triton::tools::getStrEnv("TRITON_LLVM_DEBUG_ONLY");
            !debugOnly.empty()) {
          llvm::SmallVector<std::string, 3> storage;
          llvm::SmallVector<const char *, 3> debugTypes =
              parseCommaSeparatedValues(debugOnly, storage);
          ::llvm::DebugFlag = true;
          using namespace llvm;
          setCurrentDebugTypes(debugTypes.data(), debugTypes.size());
        }

        bool haveTiming = ::triton::tools::getBoolEnv("MLIR_ENABLE_TIMING");
        if (haveTiming) {
          self.enableTiming();
        }

        // setting up diagnostics
        bool showOperations = false, showStacktraces = false,
             showRemarks = false, showWarnings = false;

        if (auto enableDiagnostics =
                triton::tools::getStrEnv("MLIR_ENABLE_DIAGNOSTICS");
            !enableDiagnostics.empty()) {
          llvm::SmallVector<std::string, 3> storage;
          parseCommaSeparatedValues(enableDiagnostics, storage);
          for (auto &str : storage) {
            if (str == "warnings") {
              showWarnings = true;
            } else if (str == "remarks") {
              showRemarks = true;
            } else if (str == "stacktraces") {
              showStacktraces = true;
            } else if (str == "operations") {
              showOperations = true;
            }
            // we show errors by default, so no need to set it
          }
        }

        DiagnosticSeverity minSeverity = showWarnings
                                             ? DiagnosticSeverity::Warning
                                             : DiagnosticSeverity::Error;
        minSeverity = showRemarks ? DiagnosticSeverity::Remark : minSeverity;

        TritonSourceMgrDiagnosticHandler diagHandler(context, minSeverity);

        context->printOpOnDiagnostic(showOperations);
        context->printStackTraceOnDiagnostic(showStacktraces);
        if (showStacktraces) {
          context->disableMultithreading();
        }
        if (failed(self.run(mod.getOperation())))
          throw std::runtime_error("PassManager::run failed");
      });
}

void init_triton_env_vars(py::module &m) {
  m.def("get_cache_invalidating_env_vars",
        []() -> std::map<std::string, std::string> {
          std::map<std::string, std::string> ret;
          for (const auto &envVar : CACHE_INVALIDATING_ENV_VARS) {
            auto strVal = triton::tools::getStrEnv(envVar);
            if (strVal.empty())
              continue;
            auto boolV = triton::tools::isEnvValueBool(strVal);
            if (boolV.has_value())
              ret[envVar] = boolV.value() ? "true" : "false";
            else
              ret[envVar] = strVal;
          }
          return ret;
        });
}
