#!/usr/bin/env python

"""Tests for `pysteps_postprocessor_diagnostics_prtype` package."""


def test_plugins_discovery():
    """It is recommended to at least test that the plugin modules provided by the plugin are
    correctly detected by pysteps. For this, the tests should be ran on the installed
    version of the plugin (and not against the plugin sources).
    """

    from pysteps.io import interface as io_interface
    from pysteps.postprocessing import interface as pp_interface

    plugin_type = "postprocessor"
    if plugin_type == "importer":
        new_importers = ["postprocessor_diagnostics_prtype"]
        for importer in new_importers:
            assert importer.replace("import_", "") in io_interface._importer_methods

    elif plugin_type == "postprocessor":
        new_postprocessors = ["postprocessor_diagnostics_prtype"]
        for postprocessor in new_postprocessors:
            postprocessor.replace("postprocessors_", "")
            if postprocessor.startswith("diagnostics"):
                assert postprocessor in pp_interface._diagnostics_methods
            elif postprocessor.startswith("ensemblestats"):
                assert postprocessor in pp_interface._ensemblestats_methods


def test_prtype_discovery():
    """ Check if the present plugin is correctly detected by pysteps. For this,
    the tests should be ran on the installed version of the plugin (and not
    against the plugin sources).
    """
    from pysteps.postprocessing import interface as pp_interface
    assert "prtype" in pp_interface._diagnostics_methods

