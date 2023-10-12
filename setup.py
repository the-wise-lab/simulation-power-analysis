from setuptools import setup, find_packages

setup(
    name='simulation_power_analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        # add any other dependencies here
    ],
    author='Toby Wise',
    description='A package for simulation-based power analysis',
    url='https://github.com/the-wise-lab/simulation-power-analysis',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
