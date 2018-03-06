# catmaid-interface
Python tools for interacting with CATMAID projects

This is a slowly increasing toolbox for extracting CATMAID (http://www.catmaid.org) annotations, typically EM reconstructions of neurons and eventually visualizing and doing various kinds of analysis of neuronal morphology, connectivity, and connections between the two. See [Quantitative neuroanatomy for connectomics in Drosophila](https://elifesciences.org/content/5/e12059) for more details of what sort of things will eventually be possible.

Currently it needs the following python packages through your favorite package manager:
* json - for parsing and handling the JSON files that CATMAID trades in.
* requests - for happily interacting with a CATMAID server.

Additionally, if you want to do analysis in Python, you'll need at least
* scipy - For quickly handling graphs using sparse matrices.

If you'd rather use Matlab, you will need to install
* matlab-json — To quickly parse JSON files. [Download here](https://github.com/christianpanton/matlab-json).

A similar tool, [nat](https://github.com/jefferis/rcatmaid), has been developed by Greg Jefferis for R.
