from setuptools import setup, find_packages

setup(
    name="qlora-llama-support",
    version="0.1.0",
    description="QLoRA fine-tuned Llama for customer support",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.40.0",
    ]
)
