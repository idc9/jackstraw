from setuptools import setup, find_packages

# def readme():
#     with open('README.rst') as f:
#            return f.read()

install_requires = ['numpy', 'sklearn', 'scipy', 'statsmodels']

setup(name='jackstraw',
      version='0.0.1',
      description='Implements jackstraw.',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
