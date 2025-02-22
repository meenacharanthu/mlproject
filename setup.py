from setuptools import find_packages, setup

def get_requirements(file_path: str) -> list:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('/n','') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
            
    return requirements

        
        
    


setup(
    name='mlproject',
    version='0.0',
    author='Meenakshi',
    author_email='meenacharanthu02@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)
    
