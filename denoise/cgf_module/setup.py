from setuptools import setup, Extension

extension = Extension(
    name='cgf', 
    sources=['src/cgf.c', 'src/brent.c', 'src/frprmn.c', 'src/linmin.c', 'src/mnbrak.c', 'src/nrutil.c'],
    include_dirs=['src'],
    # extra_compile_args=['-O3']
    # extra_compile_args=['-ggdb']
)

setup(name="cgf",
      version="1.0.0",
      description="cgf Module",
      ext_modules=[extension])
