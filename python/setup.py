from setuptools import setup, find_packages


setup(
  name = 'naiveautoml',
  packages = find_packages(exclude=["test"]),
  version = '0.1.1',
  license='MIT',
  description = 'Fast and Timeout-Free Automated Machine Learning for Multi-Class classification, Multi-Label classification, and regression.',
  author = 'Felix Mohr',
  author_email = 'mail@felixmohr.de',
  url = 'https://github.com/fmohr/naiveautoml',
  download_url = 'https://github.com/fmohr/naiveautoml/archive/refs/tags/v0.0.27.tar.gz',
  keywords = ['AutoML', 'sklearn', 'naive', 'simple', 'multi-class', 'multi-label', 'regression', 'no timeouts'],
  install_requires=[
          'numpy',
          'pandas',
          'scikit-learn==1.4',
          'scikit-multilearn==0.2.0',
          'configspace<0.7.1',
          'scipy',
          'pynisher',
          'psutil',
          'tqdm',
          'parameterized',
          'openml',
          'lccv'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
  ],
  package_data={'': ['searchspace-classification.json', 'searchspace-regression.json']},
  include_package_data=True
)
