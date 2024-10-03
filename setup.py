from setuptools import setup

setup(
  name = 'surveypy',
  version = '0.0.1',
  author = 'Ha Hoang Hao',
  packages = ['surveypy'],
  description = 'Analysis data in Survey - Market Research',
  install_requires=[
      'python-pptx==1.0.2',
      'pyreadstat',
      'pandas==2.1.4'
  ]
)