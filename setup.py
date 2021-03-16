import os, sys

################################################################
# boiler plate
################################################################

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

# root = path.abspath(path.dirname(__file__))

from webbpsf_ext.version import __version__
version = __version__

# RELEASE = 'dev' not in version
# if not RELEASE:
#     version += get_git_devstr(False)

################################################################
# shortcuts for publishing, tagging, testing
################################################################

if sys.argv[-1] == 'publish':
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_wheel upload")
    print("You probably want to also tag the version now:")
    print("  python setup.py tag")
    sys.exit()

# Requires .pypirc
#[pypitest]
#repository: https://test.pypi.org/legacy/
#username: jarron
#password: ****
if sys.argv[-1] == 'pubtest':
    os.system("python setup.py sdist upload -r pypitest")
    os.system("python setup.py bdist_wheel upload -r pypitest")
    sys.exit()
    
if sys.argv[-1] == 'tag':
    os.system("git tag -a v%s -m 'Release %s'" % (version, version))
    os.system("git push origin v%s" % (version))
    sys.exit()


if sys.argv[-1] == 'test':
    test_requirements = [
        'pytest',
        'coverage'
    ]
    try:
        modules = map(__import__, test_requirements)
    except ImportError as e:
        err_msg = e.message.replace("No module named ", "")
        msg = "%s is not installed. Install your test requirements." % err_msg
        raise ImportError(msg)
        
    os.system('py.test')
    sys.exit()

################################################################
# Package Information
################################################################

# Get the long description from the README and HISTORY files
# with open('README.rst') as readme_file:
#     readme = readme_file.read()
# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = [
    'Click>=6.0',
    'numpy>=1.13.0',
    'matplotlib>=3.0.0',
    'scipy>=1.0.0',
    'pysynphot>=0.9.7',
    'poppy>=0.9.1',
    'webbpsf>=0.9.1',
    'tqdm>4',
]

setup_requirements = ['pytest-runner', ]
test_requirements = ['pytest', ]

setup(
    name='webbpsf_ext',

    # Versions should comply with PEP440.
    version=version,

    description="WebbPSF Extensions",
    long_description="",

    # The project's main homepage.
    url='https://github.com/JarronL/webbpsf_ext',
 
    # Author details
    author='Jarron Leisenring',
    author_email='jarronl@email.arizona.edu',
    license='MIT license',
    keywords='jwst psf nircam miri simulation',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        #'Programming Language :: Python :: 2',
        #'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    packages=find_packages(include=['webbpsf_ext']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=requirements,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    #extras_require={
    #    'dev': ['check-manifest>=0.34', 'lxml>=3.6.4', 'pytest>=3.0.2'],
    #},

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #package_data={
    #    'pynrc': ['package_data.dat'],
    #},

    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
)
