
import argparse
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

def BashCommand(command):
  print(command)
  process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
  _, _ = process.communicate()


def SetupTosaRepo(path):
  os.chdir(path)

  referenceDir = path + "/reference_model"
  serializationDir = referenceDir + "/serialization"

  # Checkout the repo.
  BashCommand(gitCheckoutCmd)

  # Initialize the deps.
  referenceDir = path + "/reference_model"
  os.chdir(referenceDir)
  BashCommand(gitInitCmd)
  BashCommand(gitUpdateCmd)

  os.chdir(serializationDir)
  BashCommand("flatc --python tosa.fbs")

  # Build the contents.
  builddir = referenceDir + "/build"
  os.makedirs(builddir, exist_ok=True)
  os.chdir(builddir)
  BashCommand(cmakeInitCmd)
  BashCommand(cmakeBuildCmd)

  os.chdir(referenceDir)
  BashCommand(tosaTestsCmd)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Setup tosa reference folder')
  parser.add_argument('--path', metavar='p', type=str, help='path for setup', required=True)
  args = parser.parse_args()

  SetupTosaRepo(args.path)


