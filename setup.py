from setuptools import setup


setup(name='insolver',
      version='0.1',
      description='Mindset insurance scoring ',
      url='https://github.com/MindSetLib/MS-InsuranceScoring',
      author='Mindset',
      author_email='request@mind-set.ru',
      license='MIT',
      packages=['insolver'],
      install_requires=[
            'numpy',
            'pandas',
      ],
      zip_safe=False)