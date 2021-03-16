from setuptools import find_packages, setup

setup(
    name="deep-rlsp",
    version=0.1,
    description="Learning What To Do by Simulating the Past",
    author="David Lindner, Rohin Shah, et al",
    python_requires=">=3.7.0",
    url="https://github.com/HumanCompatibleAI/deep-rlsp",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.13",
        "scipy>=0.19",
        "sacred==0.8.2",
        "ray",
        "stable_baselines",
        "tensorflow==1.13.2",
        "tensorflow-probability==0.6.0",
        "seaborn",
        "gym",
    ],
    test_suite="nose.collector",
    tests_require=["nose", "nose-cover3"],
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
