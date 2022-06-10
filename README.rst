========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |travis| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/admk/badge/?style=flat
    :target: https://admk.readthedocs.io/
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/enricofacca/admk.svg?branch=main
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/enricofacca/admk

.. |github-actions| image:: https://github.com/enricofacca/admk/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/enricofacca/admk/actions

.. |requires| image:: https://requires.io/github/enricofacca/admk/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/enricofacca/admk/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/enricofacca/admk/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/enricofacca/admk

.. |version| image:: https://img.shields.io/pypi/v/admk.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/admk

.. |wheel| image:: https://img.shields.io/pypi/wheel/admk.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/admk

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/admk.svg
    :alt: Supported versions
    :target: https://pypi.org/project/admk

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/admk.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/admk

.. |commits-since| image:: https://img.shields.io/github/commits-since/enricofacca/admk/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/enricofacca/admk/compare/v0.0.0...main



.. end-badges

Algebraic Dynamics Monge Kantorovich solver for solution of Optimal Transport on
Graphs. It contains (part of) the work described in `Fast Iterative Solution of the Optimal Transport Problem on Graphs <https://doi.org/10.1137/20M137015X>`_. Consider citying this paper if you find the code inside this repository useful.



* Free software: MIT license

Installation
============

::

    pip install admk

You can also install the in-development version with::

    pip install https://github.com/enricofacca/admk/archive/main.zip


Documentation
=============


https://admk.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
