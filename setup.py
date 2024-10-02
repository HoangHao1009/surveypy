from setuptools import setup

setup(
  name = 'surveypy',
  version = '0.0.1',
  author = 'Ha Hoang Hao',
  packages = ['surveypy'],
  description = 'Analysis data in Survey - Market Research',
  install_requires=[
      'python-pptx',
      'pyreadstat',
      'pandas==2.2.2'
  ]
)