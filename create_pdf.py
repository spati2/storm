#
#   Usage: python createPDF.py hw1/baseline.py csc710sbse:hw1:VivekNair:vnair2
#

from __future__ import division
import sys,os
sys.dont_write_bytecode = True

def convert_to_pdf(filename, destination, comment = ""):
  if(os.path.isfile(filename)== False):
    print "Check your file path"
    return
  command1 = "a2ps --center-title=\""+str(comment)+"\" -qr1gC -f4 -o ./temp.ps " + str(filename)
  print command1
  os.system(command1)
  command2 = "ps2pdf ./temp.ps destination"
  print command2
  os.system(command2)
  os.system("rm -f temp.ps")


def get_files():
  import fnmatch
  matches = []
  for root, dirnames, filenames in os.walk('.'):
    for filename in fnmatch.filter(filenames, '*.py'):
      source =  os.path.join(root, filename)
      destination = "./pdf" + source[1:]
      print source
      print destination
      convert_to_pdf(source, destination)

if __name__ == '__main__':
  get_files();


