from module.utils import initiateFileSystem, subselectDf
from module.getters import getHeadTwitchDf
from module.histogram import headTwitchHistogram



######## INIT ##########
# Start by checking filesystem has all the folders necessary for read/write operations (cache) or create them otherwise
initiateFileSystem()
