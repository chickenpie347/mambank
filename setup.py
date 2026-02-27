from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mambank",
    version="0.1.0-alpha",
    description="MemBank: Pointer-based neural activation memory for open-source LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="chickenpie347",
    author_email="98farhan94@gmail.com",
    license="MIT",
    url="https://github.com/chickenpie347/mambank",
    packages=find_packages(exclude=["tests*", "benchmarks*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "retrieval": ["faiss-cpu>=1.7"],
        "hf":        ["transformers>=4.35", "torch>=2.0"],
        "full":      ["faiss-cpu>=1.7", "transformers>=4.35", "torch>=2.0"],
        "dev":       ["pytest>=7.0", "faiss-cpu>=1.7"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm memory activation neural pointer hgns retrieval rag",
    entry_points={
        "console_scripts": [
            "mambank-bench=benchmarks.benchmark:main",
        ]
    },
)
