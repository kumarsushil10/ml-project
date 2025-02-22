from setuptools import setup, find_packages

def get_requirements(filename):
    with open(filename, "r") as f:
        requirements = [
            line.strip() for line in f.readlines() 
            if line.strip() and not line.startswith("#")
        ]
    return requirements  

setup(
    author='Sushil Kumar',
    author_email = 'kmrsushil10@gmail.com',
    name='ML project',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)