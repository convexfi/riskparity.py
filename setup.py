from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

__version__ = "0.2"

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/riskparityportfolio*")
    sys.exit()

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'riskparityportfolio.vanilla',
        ['riskparityportfolio/vanilla.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
#        extra_link_args=["-stdlib=libc++"],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    if sys.platform == 'darwin':
        flags = ['-std=c++14', '-std=c++11']
    else:
        flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }
    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.9']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        # third-party libraries flags
        localincl = "third-party"
        if not os.path.exists(os.path.join(localincl, "eigen_3.3.7", "Eigen",
                                           "Core")):
            raise RuntimeError("couldn't find Eigen headers")
        include_dirs = [
            os.path.join(localincl, "eigen_3.3.7"),
        ]
        for ext in self.extensions:
            ext.include_dirs = include_dirs + ext.include_dirs
        # run standard build procedure
        build_ext.build_extensions(self)
setup(
    name='riskparityportfolio',
    version=__version__,
    author='Ze Vinicius & Dani Palomar',
    author_email='jvmirca@gmail.com',
    url='https://github.com/dppalomar/riskparity.py',
    description='Blazingly fast design of risk parity portfolios',
    license='MIT',
    package_dir={'riskparityportfolio' : 'riskparityportfolio'},
    packages=['riskparityportfolio'],
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4', 'numpy', 'jaxlib', 'jax', 'quadprog', 'tqdm'],
    setup_requires=['pybind11>=2.4', 'numpy', 'jaxlib', 'jax', 'quadprog', 'tqdm'],
    cmdclass={'build_ext': BuildExt},
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Financial and Insurance Industry',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.0',
    ],
    zip_safe=False,
    include_package_data=True,
)
