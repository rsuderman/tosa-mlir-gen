
import numpy as np
import os as os
import shutil as shutil
import subprocess as subprocess
import tempfile as tmpfile


gitCheckoutCmd = "git clone http://git.mlplatform.org/tosa/reference_model.git"
gitInitCmd = "git submodule init"
gitUpdateCmd = "git submodule update"

cmakeInitCmd = "cmake -G Ninja ..  -DCMAKE_C_COMPILER=clang-9 -DCMAKE_CXX_COMPILER=clang++-9"
cmakeBuildCmd = "cmake --build ."

tosaTestsCmd = "python verif/tosa_verif_build_tests.py"
tosaResultsCmd = "python verif/tosa_verif_run_ref.py"

# tmpdir = tmpfile.mkdtemp()
tmpdir = "/tmp/tmpevjmou0o"
referenceDir = tmpdir + "/reference_model"
builddir = referenceDir + "/build"
testRootDir = tmpdir + "/reference_model/vtest"

debug = False

def BashCommand(command):
  print(command)
  process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
  _, _ = process.communicate()


def SetupTosaRepo():
  os.chdir(tmpdir)

  # Checkout the repo.
  BashCommand(gitCheckoutCmd)

  # Initialize the deps.
  os.chdir(referenceDir)
  BashCommand(gitInitCmd)
  BashCommand(gitUpdateCmd)

  # Build the contents.
  os.mkdir(builddir)
  os.chdir(builddir)
  BashCommand(cmakeInitCmd)
  BashCommand(cmakeBuildCmd)

def TosaElementTypeToMlirType(dtype):
  if np.float32:
    return "f32"
  if np.int8:
    return "i8"
  if np.int16:
    return "i16"
  if np.int32:
    return "i32"
  raise Exception('Dtype not found: ' + str(dtype))

def TosaTypeToMlirType(shape, dtype):
  etype = TosaElementTypeToMlirType(dtype)
  shapeStr = "x".join([str(d) for d in shape])
  mlirType = shapeStr + "x" + etype
  return mlirType

def GenerateMlirArray(arr):
  if (len(arr.shape) == 0):
    return str(arr)
  l = []
  for v in arr:
    l.append(GenerateMlirArray(v))
  return "[" + ", ".join(l) + "]"
  

def GenerateMlirAttribute(arr):
  mlirType = TosaTypeToMlirType(arr.shape, arr.dtype)
  mlirArr = GenerateMlirArray(arr)
  const = "constant dense<{}> : tensor<{}>".format(mlirArr, mlirType)
  print(const)
  return const


def GenerateMlirInput(tensorPath):
  arr = np.load(tensorPath)
  GenerateMlirAttribute(arr)

def GenerateTosaTests():
  os.chdir(referenceDir)
  # BashCommand(tosaTestsCmd)

  # Get the list of every test available.
  allTests = []
  operations = os.listdir(testRootDir)
  for operation in operations:
    operationDir = testRootDir + "/" + operation
    for test in os.listdir(operationDir):
      allTests.append(operation + "/" + test)

  # Should loop through this.....
  # test = allTests[0]
  test = "add/add_6_float"
  print(test)
  testDir = testRootDir + "/" + test
  BashCommand(tosaResultsCmd + " -t " + test)
  files = os.listdir(testDir)

  tosa = sorted([f for f in files if f == 'test.tosa'])
  inputs = sorted([f for f in files if "input-" in f and ".npy" in f])
  results = sorted([f for f in files if "result-" in f and ".npy" in f])

  if len(tosa) != 1:
    return

  for t in inputs:
    inputPath = testDir + "/" + t
    GenerateMlirInput(inputPath)

  for result in results:
    print(testDir + "/" + result)


def GenerateMlirTests():
  pass


# os.mkdir(tmpdir)
# SetupTosaRepo()
GenerateTosaTests()

# Generate the reference models.
# if not debug:
#   BashCommand(buildTestsCmd)


