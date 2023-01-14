#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['Pillow>=7.0,<10']

setup_requirements = []

test_requirements = ['pytest']

setup(
    author="Loune Lam, Glenn Gribble",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="SVG Plugin for Pillow",
    entry_points={},
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='pillow_svg',
    name='pillow_svg',
    packages=find_packages(include=['pillow_svg', 'pillow_svg.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/gribbg/pillow_svg',
    version='0.1.0',
    zip_safe=False,
)
