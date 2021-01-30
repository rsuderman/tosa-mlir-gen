
import argparse
import os
from convert_tosa_to_mlir import GenerateTosaToMlir
from generate_mlir_cpu_runner_test import GenerateTosaTest

def DoTask(inDir, refDir, outTest, mode):
  outMlir = os.path.join(inDir, "test.mlir")

  try:
    GenerateTosaToMlir(inDir, refDir, outMlir)
  except:
    pass

  if (not os.path.exists(outMlir)):
    print("Failed to translate flatbuffer: ", outMlir.split("/")[-2])
    return

  os.makedirs(os.path.dirname(outTest), exist_ok=True)

  GenerateTosaTest(outMlir, inDir, refDir, outTest, mode)
  if (not os.path.exists(outTest)):
    print("Failed to create test: ", outMlir.split("/")[-2])

  print("Success: ", outMlir.split("/")[-2])

def GetTasks(referenceDir, outDir, op=None):
  vtestDir = os.path.join(referenceDir, "vtest")
  tests = []
  ops = [op] if op is not None else os.listdir(vtestDir)

  for op in ops:
    opDir = os.path.join(vtestDir, op)
    for test in os.listdir(opDir):
      testOut = os.path.join(outDir, test + ".mlir")
      testDir = os.path.join(opDir, test)
      tests.append((testOut, testDir))
  return tests

def GetDebugTasks(referenceDir, outDir):
  vtestDir = os.path.join(referenceDir, "vtest")
  testOut = os.path.join(outDir, "abs_6_float.mlir")
  testDir = os.path.join(vtestDir, "abs/abs_6_float/")
  return [(testOut, testDir)]


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Batch function to generate test')
  parser.add_argument('--reference-dir', metavar='p', type=str, help='reference-dir-path', required=True)
  parser.add_argument('--out-dir', metavar='p', type=str, help='out-dir-path', required=True)
  parser.add_argument('--op', type=str, help='operation to rune', default=None)
  parser.add_argument('--mode', metavar='m', type=str, help='test mode', required=True, choices=['cpu-runner', 'iree'])

  args = parser.parse_args()
  tasks = GetTasks(args.reference_dir, args.out_dir, args.op)
  # tasks = GetDebugTasks(args.reference_dir, args.out_dir)
  for task in tasks[:100]:
    (outTest, inMlir) = task
    DoTask(inMlir, args.reference_dir, outTest, args.mode)


