from module.utils import initiateFileSystem, subselectDf
from module.getters import getRawDf
from module.histogram import singleHistogram, behaviorHistogram



######## INIT ##########
# Start by checking filesystem has all the folders necessary for read/write operations (cache) or create them otherwise
initiateFileSystem()
