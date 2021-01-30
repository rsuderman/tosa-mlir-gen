
import argparse
import numpy as np
import os as os
import shutil as shutil
import subprocess as subprocess
import tempfile as tmpfile


tosaResultsCmd = "python verif/tosa_verif_run_ref.py"

cpuRunnerHeader = ("// RUN: mlir-opt %s --tosa-to-linalg-on-tensors -convert-elementwise-to-linalg -std-bufferize -tensor-constant-bufferize -linalg-bufferize -tensor-bufferize -func-bufferize -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm | \\\n"
"// RUN: mlir-cpu-runner -e test_main -entry-point-result=void -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext |\\\n"
"// RUN: FileCheck %s\n\n"
"func private @print_memref_f32(%ptr : tensor<*xf32>)\n"
"func private @print_memref_f64(%ptr : tensor<*xf64>)\n"
"func private @print_memref_i8 (%ptr : tensor<*xi8>)\n"
"func private @print_memref_i16(%ptr : tensor<*xi16>)\n"
"func private @print_memref_i32(%ptr : tensor<*xi32>)\n"
"func private @print_memref_i64(%ptr : tensor<*xi64>)\n"
)

def BashCommand(command):
  process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
  _, _ = process.communicate()


class MlirTestPrinter:
  def __init__(self, mode):
    self.var = 0
    if mode != 'iree' and mode != 'cpu-runner':
      raise Exception("invalid test printer mode: " + str(mode))
    self.mode = mode

  def NewVar(self):
    newVar = self.var
    self.var += 1
    return newVar

  def TosaElementTypeToMlirType(self, dtype):
    if dtype == np.bool:
      return "i1"
    if dtype == np.float32:
      return "f32"
    if dtype == np.int8:
      return "i8"
    if dtype == np.int16:
      return "i16"
    if dtype == np.int32:
      return "i32"
    raise Exception('Dtype not found: ' + str(dtype))
  
  def TosaTypeToMlirType(self, arr):
    shape = arr.shape
    dtype = arr.dtype
    etype = self.TosaElementTypeToMlirType(dtype)
    shapeStr = "x".join([str(d) for d in shape])
    mlirType = shapeStr + "x" + etype
    return "tensor<{}>".format(mlirType)
  
  def TosaTypeToFlatMlirType(self, arr):
    dtype = arr.dtype
    etype = self.TosaElementTypeToMlirType(dtype)
    return "tensor<{}x{}>".format(arr.size, etype)
  
  def GenerateMlirArray(self, arr):
    if (len(arr.shape) == 0):
      return str(arr)
    l = []
    for v in arr:
      l.append(self.GenerateMlirArray(v))
    return "[" + ", ".join(l) + "]"
  
  def GenerateMlirConst(self, arr):
    mlirType = self.TosaTypeToMlirType(arr)
    mlirArr = self.GenerateMlirArray(arr)
    return "constant dense<{}> : {}".format(mlirArr, mlirType)
  
  def GenerateCall(self, mainVar, inputVars, inputs, results):
    callLine = []
    callLine.append("  %{} = call @main(".format(mainVar))
    callLine.append(", ".join("%" + str(i) for i in inputVars))
    callLine.append(") : (")
    callLine.append(", ".join([self.TosaTypeToMlirType(inp) for inp in inputs]))
    callLine.append(") -> (")
    callLine.append(", ".join([self.TosaTypeToMlirType(res) for res in results]))
    callLine.append(")")
    return "".join(callLine)

  def GenerateHeader(self):
    if self.mode == 'iree':
      return ""
    return cpuRunnerHeader
  
  def GenerateCpuRunnerCheck(self, funcVar, expected):
    lines = []
    etype = self.TosaElementTypeToMlirType(expected.dtype)

    # TODO(suderman): Update check to include values/shape.
    lines.append("// CHECK: Unranked Memref base@ = ")

    reshapeVar = funcVar
    if (len(expected.shape) != 1):
      reshapeVar = self.NewVar()
      mapping = ", ".join([("d" + str(d)) for d in range(len(expected.shape))])
      lines.append("  %{} = linalg.tensor_reshape %{} [affine_map<({}) -> ({})>] : {} into {}".format(
        reshapeVar, funcVar, mapping, mapping, self.TosaTypeToMlirType(expected), self.TosaTypeToFlatMlirType(expected)))

    castVar = self.NewVar()
    lines.append("  %{} = tensor.cast %{} : {} to tensor<*x{}>".format(
      castVar, reshapeVar, self.TosaTypeToFlatMlirType(expected), etype))
    lines.append("  call @print_memref_{}(%{}) : (tensor<*x{}>) -> ()".format(etype, castVar, etype))
    return "\n".join(lines)

  def GenerateIreeCheck(self, funcVar, expected):
    lines = []
    mlirType = self.TosaTypeToMlirType(expected)
    mlirArr = self.GenerateMlirArray(expected)
    etype = self.TosaElementTypeToMlirType(expected.dtype)
    if etype in ['f32', 'f64']:
      lines.append("  check.expect_almost_eq_const(%{}, dense<{}> : {}) : {}".format(funcVar, mlirArr, mlirType, mlirType))
    else:
      lines.append("  check.expect_eq_const(%{}, dense<{}> : {}) : {}".format(funcVar, mlirArr, mlirType, mlirType))

    return "\n".join(lines)

  def GenerateResultCheck(self, funcVar, result):
    if self.mode == 'iree':
      return self.GenerateIreeCheck(funcVar, result)
      
    return self.GenerateCpuRunnerCheck(funcVar, result)

  def GenerateTestMain(self, inputs, results):
    lines = []

    # Initialize the main testing function.
    if self.mode == 'iree':
      lines.append("func @test_main() attributes { iree.module.export } {")
    else:
      lines.append("func @test_main() {")
  
    inputVars = []

    # Add all the input testing constants.
    for arr in inputs:
      newVar = self.NewVar();
      inputVars.append(newVar)
      lines.append("  %{} = {}".format(newVar, self.GenerateMlirConst(arr)))
    lines.append("")
  
    # Call the function being tested.
    funcVar = self.NewVar()
    funcVarWithCount = funcVar if len(results) < 2 else "{}:{}".format(funcVar, str(len(results)))
    lines.append(self.GenerateCall(funcVarWithCount, inputVars, inputs, results))
  
    # Reshape, cast, and print each result type
    for idx, result in enumerate(results):
      lines.append("")
      resultVar = funcVar if len(results) < 2 else "{}#{}".format(funcVar, str(idx))
      lines.append(self.GenerateResultCheck(resultVar, result))

    lines.append("  return")
    lines.append("}")
    return "\n".join(lines)
  
  def GenerateTestFile(self, mlir_file, input_dir):
    files = os.listdir(input_dir)
    tosaMlir = mlir_file
    inputs = sorted([f for f in files if "input-" in f and ".npy" in f])
    results = sorted([f for f in files if "result-" in f and ".npy" in f])
  
    inputs = [np.load(input_dir + "/" + t) for t in inputs]
    results = [np.load(input_dir + "/" + t) for t in results]

    tosaMlirContents = open(tosaMlir).read()

    mainLines = []
    mainLines.append(self.GenerateHeader())
    mainLines.append(tosaMlirContents)
    mainLines.append("")
    mainLines.append(self.GenerateTestMain(inputs, results))

    return "\n".join(mainLines)

def GenerateTosaTest(mlirFile, inputDir, refDir, outputFile, mode):
  os.chdir(refDir)

  # Should loop through this.....
  BashCommand(tosaResultsCmd + " -t " + inputDir)

  printer = MlirTestPrinter(mode)
  ir = printer.GenerateTestFile(mlirFile, inputDir)

  if outputFile:
    mlirFile = open(outputFile, "w")
    mlirFile.write(ir)
    mlirFile.write("\n")
    mlirFile.close()
  else:
    print(ir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Convert create an MLIR cpu-runner test')
  parser.add_argument('--input-mlir-file', metavar='i', type=str, help='path for input mlir file', required=True)
  parser.add_argument('--input-path', metavar='i', type=str, help='path for input path', required=True)
  parser.add_argument('--tosa-reference-library', metavar='i', type=str, help='path for input file', required=True)
  parser.add_argument('--output-file', metavar='i', type=str, help='path for output file')
  parser.add_argument('--mode', metavar='m', type=str, help='mode for generating test', choices=['cpu-runner', 'iree'])
  args = parser.parse_args()

  mlirFile = args.input_mlir_file
  inputDir = args.input_path
  referenceDir = args.tosa_reference_library
  outputFile = args.output_file

  GenerateTosaTest(mlirFile, inputDir, outputFile)

