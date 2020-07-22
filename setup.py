from setuptools import setup, find_packages


setup(
    name="Capstone", use_scm_version=True, install_requires=['pydicom', 'torchsummary'], packages=find_packages(),
)
