# data_pipeline

This folder is supposed to contain a well documented data-processing pipeline
from raw Ca-images (from neuronal data) to a thorough receptive field analysis.

### Thoughts:
currently, CaImAn requires (I think) tiff files to perform memory efficient
creation of memmap-files. I suppose most data comes in tiff/format, though saving
in zarr/hdf5 might be way more efficient (for both, reading and writing).
CHECK: are hdf5 files already implemented? what is the proper syntax then?
if not: possible to implement?

### structure of the data-processing
The analysis pipeline consists of:

|
| I. preprocessing
| - I.1. median filter
| - I.2. alignment / motion correction
| - I.3. casting to proper type (memmap)
|
| II. image analysis
| - II.1. [CaImAn] neuron detection & trace extraction
| - II.2. [CaImAn] component evaluation
| - II.3. saving relevant data
|
| III. analysis
| - III.1. ROI matching
| - III.2. place field detection
| - III.3. everything else
