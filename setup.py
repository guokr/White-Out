from setuptools import setup, find_packages
import sys

if sys.version_info<(3,):
    sys.exit("Sorry, Python 3 is required for Caver")

dep_private = []
dep_pypi = []

def get_requirements():
    with open("requirements.txt", "r") as f:
        reqs = [l for l in f.read().splitlines() if l]
        for _ in reqs:
            if _.startswith("git"):
                dep_private.append(_)
            else:
                dep_pypi.append(_)

setup(
    name="white-out",
    version="0.1",
    description="auto correction lib",
    # long_description=readme,
    author='Guokr Inc.',
    author_email='jinyang.zhou@guokr.com',
    url="https://github.com/guokr/White-Out",
    packages=find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ),
    entry_points={
        'console_scripts': [
            # 'trickster_train=trickster::train',
        ]
    },
    install_requires=dep_pypi,
    dependency_links=dep_private
)
