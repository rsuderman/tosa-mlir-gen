
import argparse
import flatbuffers
import importlib.util
import numpy as np
import os as os
import shutil as shutil
import subprocess as subprocess
import tempfile as tmpfile
import sys


class RegisterMap():
  def __init__(self):
    self._argcount = 0
    self._varcount = 0
    self._stack = [{}]

  # def Push(self):
  #   self._stack.append(self._stack[-1].deepcopy())

  # def Pop(self):
  #   self._stack = self._stack[:-1]

  def AddArg(self, name):
    newarg = "%arg{}".format(self._argcount)
    self._stack[-1][name] = newarg
    self._argcount = self._argcount + 1
    return newarg

  def AddVar(self, name):
    if not isinstance(name, list):
      newvar = "%{}".format(self._varcount)
      self._stack[-1][name] = newvar
      self._varcount = self._varcount + 1
      return newvar
    
    if (len(name) < 2):
      newvar = "%{}".format(self._varcount)
      self._stack[-1][name[0]] = newvar
      self._varcount = self._varcount + 1
      return newvar

    for i, n in enumerate(name):
      newvar = "%{}#{}".format(self._varcount, i)
      self._stack[-1][n] = newvar

    vardecl = "%{}:{}".format(self._varcount, len(name))
    self._varcount = self._varcount + 1
    return vardecl

  def Get(self, name):
    return self._stack[-1][name]

  def GetVars(self, lst):
    return [self.Get(v) for v in lst]

  def __getitem__(self, key):
      return self.Get(key)

class MlirPrinter():
  def __init__(self, path):
    buf = open(path, 'rb').read()
    self.graph = TosaGraph.TosaGraph.GetRootAsTosaGraph(buf, 0)

    self.OpMap = {}
    self.blockMap = self.GetBlockMap(self.graph)
    self.mainBlock = self.blockMap[b'main']
    self.registerMap = self.InitRegisterMap(self.mainBlock, 0)

    self.tensorTypeMap = {}
    self.UpdateTensorTypeMap(self.mainBlock)

    for key in OpEnum.Op.__dict__.keys():
      if key.startswith("__"):
        continue
      self.OpMap[OpEnum.Op.__dict__[key]] = key.lower()

  def GetMlirElementType(self, elementTy):
    if (elementTy == TosaType.DType.BOOL):
      return "i1"
    if (elementTy == TosaType.DType.UINT8):
      return "ui8"
    if (elementTy == TosaType.DType.INT4):
      return "i4"
    if (elementTy == TosaType.DType.INT8):
      return "i8"
    if (elementTy == TosaType.DType.INT16):
      return "i16"
    if (elementTy == TosaType.DType.INT32):
      return "i32"
    if (elementTy == TosaType.DType.INT48):
      return "i48"
    if (elementTy == TosaType.DType.FLOAT):
      return "f32"
    raise Exception("Unknown type: " + elementTy)

  def GetTensorType(self, tensor):
    rank = tensor.ShapeLength()
    elementTy = self.GetMlirElementType(tensor.Type())
    shape = "x".join([str(tensor.Shape(i)) for i in range(rank)])
    return "tensor<{}x{}>".format(shape, elementTy)

  def InitRegisterMap(self, block, varStart):
    registerMap = RegisterMap()
    for i in range(block.InputsLength()):
      registerMap.AddArg(block.Inputs(i))
    return registerMap


  def UpdateTensorTypeMap(self, block):
    for i in range(block.TensorsLength()):
      tensor = block.Tensors(i)
      tensorTy = self.GetTensorType(tensor)
      self.tensorTypeMap[tensor.Name()] = tensorTy

  def GetFunctionHeader(self, block):
    name = block.Name()
    typeMap = self.tensorTypeMap

    argDecl = ["{} : {}".format(self.registerMap[block.Inputs(i)], typeMap[block.Inputs(i)]) for i in range(block.InputsLength())]
    argDecl = ", ".join(argDecl)

    retTypes = [typeMap[block.Outputs(i)] for i in range(block.OutputsLength())]
    retTypes = ", ".join(retTypes)

    return "func @{}({}) -> ({}) {{".format(str(name.decode()), argDecl, retTypes)

  def GetReturnOp(self, block):
    registerMap = self.registerMap
    typeMap = self.tensorTypeMap

    retVals = [registerMap[block.Outputs(i)] for i in range(block.OutputsLength())]
    retTypes = [typeMap[block.Outputs(i)] for i in range(block.OutputsLength())]

    retVals = ", ".join(retVals)
    retTypes = ", ".join(retTypes)

    return "  return {} : {}".format(retVals, retTypes)

  def GetOperatorName(self, openum):
    opname = self.OpMap[openum]
    if opname == "unknown":
      raise "Unknown Op"
    return "\"tosa.{}\"".format(opname)
    
  def GetOperatorIR(self, operator):
    opname = self.GetOperatorName(operator.Op())
    inputs = [operator.Inputs(i) for i in range(operator.InputsLength())]
    outputs = [operator.Outputs(i) for i in range(operator.OutputsLength())]

    # Handle a placeholder special case.
    if opname == '"tosa.placeholder"':
      if not inputs:
        return ("  // placeholder() -> " + outputs[0].decode())

    # TODO(suderman): Handle attribute info.
    if operator.AttributeType() != 0 or operator.Attribute() is not None:
      raise Exception("Unhandled attribute")

    # TODO(suderman): Handle quant info.
    if operator.QuantInfoType() != 0 or operator.QuantInfo() is not None:
      raise Exception("Unhandled quantization information")

    var = self.registerMap.AddVar(outputs)

    args = ", ".join(self.registerMap.GetVars(inputs))
    argTypes = ", ".join([self.tensorTypeMap[inp] for inp in inputs])
    retTypes = ", ".join([self.tensorTypeMap[out] for out in outputs])

    return "  {} = {}({}) : ({}) -> ({})".format(var, opname, args, argTypes, retTypes)

  def GetBlockIR(self, block):
    self.UpdateTensorTypeMap(block)
    lines = []
    for i in range(block.OperatorsLength()):
      operator = block.Operators(i)
      lines.append(self.GetOperatorIR(operator))
   
    return lines

  def GetBlockMap(self, graph):
    blockMap = {}
    for i in range(graph.BlocksLength()):
      blockMap[graph.Blocks(i).Name()] = graph.Blocks(i)
    return blockMap


  def GetIR(self):
    blockIR = []
    blockIR.append(self.GetFunctionHeader(self.mainBlock))
    blockIR.append("\n".join(self.GetBlockIR(self.mainBlock)))
    blockIR.append(self.GetReturnOp(self.mainBlock))
    blockIR.append("}")
    blockIR.append("")
    return "\n".join(blockIR)
    
def GlobalSetup(referenceDir):
  serializationDir = referenceDir + "/serialization"
  global TosaGraph
  global TosaType
  global OpEnum
  sys.path.append(serializationDir)
  import tosa.TosaGraph as TosaGraph
  import tosa.DType as TosaType
  import tosa.Op as OpEnum

def GenerateTosaToMlir(testDir, tosaReferenceLibrary, outputFile):
  GlobalSetup(tosaReferenceLibrary)

  tosaFile = testDir + "/test.tosa"
  mlirFile = testDir + "/test.mlir"
  printer = MlirPrinter(tosaFile);
  ir = printer.GetIR()

  if outputFile:
    mlirFile = open(outputFile, "w")
    mlirFile.write(ir)
    mlirFile.write("\n")
    mlirFile.close()
  else:
    print(ir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Convert TOSA flatbuffer to MLIR')
  parser.add_argument('--input-path', metavar='i', type=str, help='path for input path', required=True)
  parser.add_argument('--tosa-reference-library', metavar='i', type=str, help='path for input file', required=True)
  parser.add_argument('--output-file', metavar='i', type=str, help='path for output file')
  args = parser.parse_args()

  GenerateTosaToMlir(args.input_path, args.tosa_reference_library, args.output_file)

