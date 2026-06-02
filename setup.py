from setuptools import setup, find_packages

setup(
    name='spine-mri-segmentation',
    version='1.0.0',
    description='Lumbar Spine MRI Segmentation System',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'SimpleITK>=2.1.0',
        'matplotlib>=3.5.0',
        'tqdm>=4.64.0',
        'PyYAML>=6.0',
        'einops>=0.5.0',
    ],
)
