from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements: List[str] = []
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = [req.strip() for req in file_obj.readlines() if req.strip()]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
    name="mlproject",
    version="0.1.0",
    author="Suryavhi",
    author_email="dassuryavhi704@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)