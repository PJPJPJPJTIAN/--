from setuptools import setup, find_packages  # 导入必要的工具

setup(
    name="eoh",  # 包的名称，安装后可用 `import eoh` 导入
    version="0.1",  # 版本号，遵循语义化版本规范
    author="MetaAI Group, CityU",  # 作者/机构信息
    description="Evolutionary Computation + Large Language Model for automatic algorithm design",  # 包的简要描述
    packages=find_packages(where='src'),  # 自动发现 `src` 目录下的所有 Python 包
    package_dir={'': 'src'},  # 指定包的根目录为 `src`（即包的代码在 src 文件夹下）
    python_requires=">=3.10",  # 要求的 Python 版本最低为 3.10
    install_requires=[  # 安装该包时需要自动安装的依赖库
        "numpy",
        "numba",
        "joblib"
    ],
    test_suite="tests"  # 测试套件的位置（指定测试代码所在的目录）
)