from setuptools import setup

setup(
  name = 'surveypy',
  version = '0.0.1',
  author = 'Ha Hoang Hao',
  packages = ['surveypy'],
  description = 'Analysis data in Survey - Market Research',
  install_requires=[
      'python-pptx',  # Add required dependencies here
      'pyreadstat',
      # 'langchain',
      # 'langchain_community',
      # 'langchain_huggingface',
      # 'langchain_chroma',
      # 'langchain_openai'
  ],
)