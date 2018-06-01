# catmaid-interface
Python tools for interacting with CATMAID projects

This is a slowly increasing toolbox for extracting CATMAID (http://www.catmaid.org) annotations, typically EM reconstructions of neurons and eventually visualizing and doing various kinds of analysis of neuronal morphology, connectivity, and connections between the two. See [Quantitative neuroanatomy for connectomics in Drosophila](https://elifesciences.org/content/5/e12059) for more details of what sort of things will eventually be possible.

Currently it needs the following python packages through your favorite package manager:
* json
* requests 
* numpy
* scipy
* matplotlib
* pandas
* plotly 
* networkx
* tqdm
* sklearn

If you'd rather use Matlab, you will need to install
* matlab-json — To quickly parse JSON files. [Download here](https://github.com/christianpanton/matlab-json).

A similar tool, [nat](https://github.com/jefferis/rcatmaid), has been developed by Greg Jefferis for R.
