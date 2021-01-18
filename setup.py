from setuptools import setup
import pkg_resources

with open("README.md", 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as requirements:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements)
    ]

setup(
    name='pyautofit',
    version='0.0.0',
    description='Might be a useful package',
    long_description=long_description,
    author='Bowen Huang',
    author_email='bowen.huang@equifax.com',
    packages=['pyautofit'],
    install_requires=install_requires
)