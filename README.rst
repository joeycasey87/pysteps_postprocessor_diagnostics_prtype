========================================
pysteps-postprocessor-diagnostics-prtype
========================================

Pysteps plugin for calculating the precipitation type of hydrometeors.


License
=======
* BSD license


Documentation
=============

This is a plugin designed for implementation alongside the pySTEPS package. This plugin contains functions which will allow for the calculation of the precipitation type of the hydrometeors present in a pySTEPS nowcast. In order to use this functionality, the user must provide a pySTEPS nowcast as well as arrays featuring the snowfall level, air temperature, and surface temperature data of the region covered by the nowcast. A digital elevation model of the region and the metadata of the data will also be required. The plugin is weather model independent and, as such, the user will have to utilize their own data importer to extract the required information from their weather model. An example data importer is provided in the docs folder which can be used to extract the required data from INCA grib files.

Installation instructions
=========================

This plugin can be installed directly from github using:

.. code-block:: console

  $ pip install git+https://github.com/joeycasey87/pysteps_postprocessor_diagnostics_prtype

Credits
=======

- This package was created with Cookiecutter_ and the `pysteps/cookiecutter-pysteps-plugin`_ project template.

.. Since this plugin template is based in the cookiecutter-pypackage template,
it is encouraged to leave the following credits to acknowledge Audrey Greenfeld's work.

- The `pysteps/cookiecutter-pysteps-plugin`_ template was adapted from the cookiecutter-pypackage_
template.

.. _cookiecutter-pypackage: https://github.com/audreyfeldroy/cookiecutter-pypackage

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`pysteps/cookiecutter-pysteps-plugin`: https://github.com/pysteps/cookiecutter-pysteps-plugin
