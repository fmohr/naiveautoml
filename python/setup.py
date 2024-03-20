from distutils.core import setup
setup(
  name = 'naiveautoml',
  packages = ['naiveautoml'],
  version = '0.0.28',
  license='MIT',
  description = 'The official package for the Naive AutoML paper',
  author = 'Felix Mohr',
  author_email = 'mail@felixmohr.de',
  url = 'https://github.com/fmohr/naiveautoml',
  download_url = 'https://github.com/fmohr/naiveautoml/archive/refs/tags/v0.0.27.tar.gz',
  keywords = ['AutoML', 'sklearn', 'naive', 'simple'],
  install_requires=[
          'numpy',
          'pandas',
          'scikit-learn>=1.4',
          'configspace<0.7.1',
          'scipy',
          'pynisher',
          'psutil',
          'tqdm',
          'parameterized',
          'openml'
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
