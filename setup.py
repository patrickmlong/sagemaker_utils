from distutils.core import setup

setup(name='sagemaker_utils',
      version='1.0dev',
      description='Python utilities for AWS Sagemaker setup',
      author='Patrick Long',
      author_email='patrick.long@gmail.com',
      packages=
      ['sagemaker_utils'],
      package_dir = {'': 'src'}
      )
