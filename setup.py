#!/usr/bin/env python

from setuptools import setup
from setuptools.command.develop import develop
import setuptools
import os
import pathlib
from pathlib import Path

class CustomDevelopCommand(develop):
    def run(self):
        super().run()
        
        # Determine the bin directory dynamically
        bin_dir = Path(sys.prefix+'/bin')
        # Path to the local scripts in your project
        local_scripts = Path(__file__).parent / "bin"
        
        # Ensure symlinks for each script
        for script in local_scripts.iterdir():
            if script.is_file():
                target = bin_dir / script.name
                if not target.exists():
                    print(f"Creating symlink: {target} -> {script.resolve()}")
                    target.symlink_to(script.resolve())


if __name__ == "__main__":
    setup(name = "naz",  # Required
          version = "0.0.1",  # Required
          description = "Normalizing flow Algorithms beyond Zero-variance training",  # Optional
          python_requires = ">=3.11",
          license_file="LICENSE",
          author= "Anarya Ray",
          author_email = "anarya.ray@northwestern.edu",
          maintainer= "Anarya Ray, Ryan Magee",
          maintainer_email = "anarya.ray@ligo.org, ryan.magee@ligo.org",
          package_dir={"": "src"},
          packages = setuptools.find_packages(where='src'),
          scripts=[]
          cmdclass={"develop": CustomDevelopCommand},)
