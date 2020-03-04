Code, documentation, data and Jupyter Notebook associated with the publication
"Taxonomist: Application Detection Through Rich Monitoring Data" for the
European Conference on Parallel Processing 2018.

The related study develops a technique named 'Taxonomist' to identify
applications running on supercomputers, using machine learning to classify known
applications and detect unknown applications. The technique uses monitoring data
such as CPU and memory usage metrics and hardware counters collected from
supercomputers. The aims of this technique include providing an alternative to
'naive' application detection methods based on names of processes and scripts,
and helping prevent fraud, waste and abuse in supercomputers.

Taxonomist uses supervised learning techniques to automatically select the most
relevant features that lead to reliable application identification. The process
involves the following steps:

1. Monitoring data is collected from every compute node in a time series format.
2. 11 statistical features are extracted over the time series (e.g. percentiles,
   minimum, maximum, mean), thus reducing storage and computation overhead.
3. A classifier is trained based on a set of labeled applications, based on a
   'one-versus-rest' version of that classifier - effectively for each
   application in the training set a separate classifier is trained to
   differentiate that application.

The dataset consists of:

**README.pdf** - user guide for the 'Taxonomist' artifact outlining installation and
instructions for using the Jupyter notebook, as well as code omissions in
notebook compared to a described in Euro-Par 2018 process.
**taxonomist.py** - Python file including a basic version of the Taxonomist
framework. The module contents can be imported for other projects.
**noteboook.html** - static HTML version of the notebook that can be viewed by a
browser.
**notebook.ipynb** - interactive Jupyter Notebook file, for operation see
README.pdf.
**data/** - Folder containing monitoring data collected from different
applications executed on Volta:
- **metadata.csv**: A csv file listing each run, the IDs of the nodes on which each
  run executed, which application was executed with which inputs, the start and
  end times and the duration of the applications.

- **timeseries.tar.bz2**: This file is removed from GitHub, but it can be
  downloaded from the FigShare link below.

- **features.hdf**: A HDF5 File containing the pre-calculated features. The
  calculation process is included in the notebook.

**requirements.txt** - list of Python packages required.
**LICENSE** - the licence under which this software is released

Files are in in openly accessible Python language (.py and ipynb), .html. pdf,
.csv, .txt .zip and Hierarchical Data Format .hdf formats.

Experimental set-up for the experiments reported in the related publication uses
Volta, a Cray XC30m supercomputer located at Sandia National Laboratories, as
well as the open source monitoring tool Lightweight Distributed Metric System
(LDMS).

The original URL for the artifact is https://figshare.com/articles/Artifact_for_Taxonomist_Application_Detection_through_Rich_Monitoring_Data/6384248
