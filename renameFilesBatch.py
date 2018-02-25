import glob, os

def rename(dir, pattern, titlePattern):
    increment = 0
    tpattern = titlePattern
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        increment = increment + 1        
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        newname = tpattern + str(increment)+ ext
        os.rename(pathAndFilename, 
                 os.path.join(dir, newname))
