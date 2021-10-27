from distutils.core import setup
setup(
  name = 'naiveautoml',         # How you named your package folder (MyLib)
  packages = ['naiveautoml'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'The official package for the Naive AutoML paper',   # Give a short description about your library
  author = 'Felix Mohr',                   # Type in your name
  author_email = 'mail@felixmohr.de',      # Type in your E-Mail
  url = 'https://github.com/fmohr/naiveautoml',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/fmohr/naiveautoml/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['AutoML', 'sklearn', 'naive', 'simple'],
  install_requires=[
          'validators',
          'beautifulsoup4',
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
)