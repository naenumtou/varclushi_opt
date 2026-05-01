
from setuptools import setup, find_packages

setup(
    name = "varclushi_opt",
    version = "0.1",
    author = "LengYi",
    author_email = "naenumtou@gmail.com",
    description = "A package for variable clustering like PROC VARCLUS in SAS",
    url = "https://github.com/naenumtou/varclushi_opt",
    packages = find_packages(),
	install_requires = [
		"pandas",
		"numpy",
		"factor-analyzer==0.3.1",
	],
    classifiers = [
		"Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
		"Operating System :: Microsoft :: Windows",
		"Operating System :: POSIX :: Linux",
		"Operating System :: Unix"
    ]
)