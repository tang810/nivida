# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Conda 环境路径
conda_env_path = '/mnt/sdb/anaconda3/envs/AnalysisData'
alpha_path = f'{conda_env_path}/lib/python3.10/site-packages'

# 收集所有必要的隐藏导入
hidden_imports = [
    # Alpha 相关
    *collect_submodules('alpha'),
    
    # FastAPI 相关
    'fastapi',
    'uvicorn',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.websockets_impl',
    'uvicorn.protocols.http.httptools_impl',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.loops.auto',
    'uvicorn.loops.asyncio',
    'starlette.middleware.trustedhost',
    
    # 数据处理相关
    'pandas',
    'numpy',
    'matplotlib',
    'matplotlib.backends.backend_agg',
    
    # 机器学习相关
    'sentence_transformers',
    'transformers',
    'torch',
    'sklearn',
    'chromadb',
    
    # Langchain 相关
    'langchain',
    'langchain.llms',
    'langchain.embeddings',
    'langchain.vectorstores',
    
    # 其他可能需要的模块
    'openmdao',
    'openaerostruct',
    'ADRpy',
    'multipart',
    'python_multipart',
]

# 收集数据文件
datas = [
    # 项目目录
    ('config', 'config'),
    ('fonts', 'fonts'),
    ('upload', 'upload'),
    ('images', 'images'),
    ('tools', 'tools'),
    
    # Alpha 模块
    (f'{alpha_path}/alpha', 'alpha'),
    
    # 可能需要的模型文件和配置文件
    ('*.py', '.'),
    ('.env', '.'),
]

# 尝试收集一些包的数据文件
try:
    datas.extend(collect_data_files('sentence_transformers'))
    datas.extend(collect_data_files('transformers'))
    datas.extend(collect_data_files('chromadb'))
except:
    pass

a = Analysis(
    ['main.py'],
    pathex=[
        alpha_path,
        conda_env_path + '/lib/python3.10/site-packages',
        '.'
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib.tests',
        'pandas.tests',
        'numpy.tests',
        'sklearn.tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# 过滤掉一些不需要的文件
a.datas = [x for x in a.datas if not (
    x[0].endswith('.pyc') or 
    x[0].endswith('.pyo') or
    '/tests/' in x[1] or
    '/__pycache__/' in x[1]
)]

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='analysis_data',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='analysis_data'
)