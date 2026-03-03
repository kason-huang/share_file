数据准备
obsutil config -i=HPUAXIGJCSQZZAWTAFGF -k=v1756rW0y5OJDF5rAfmej9uhLHOKtCJotvTGA6cZ -e=https://obs.cn-east-3.myhuaweicloud.com

obsutil cp obs://data-platform-shanghai/data/streamvln_data/datasets.zip datasets.zip 

# 150K
episode_num_3100-3199
...
episode_num_6800-6899
./obsutil sync obs://data-platform-shanghai/datatraj_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content/episode_num_3100-3199  episode_num_3100-3199
./obsutil sync obs://data-platform-shanghai/datatraj_datasets/vln/hm3d_v2_l3mvn_refine_v2_1/train/content/episode_num_3100-3199  episode_num_3100-3199



Example严重
1. 轨迹数据集 
obsutil cp obs://data-platform-shanghai/data/traj_datasets/objectnav/cloudrobo_v1_l3mvn/train/content/suzhou-room-shengwei-metacam-2025-07-09_01-13-22.json.gz suzhou-room-shengwei-metacam-2025-07-09_01-13-22.json.gz
[图片]
2. 场景集（重建）
obsutil sync obs://data-platform-shanghai/自采重建场景/suzhou-room-shengwei-metacam-2025-07-09_01-13-22 suzhou-room-shengwei-metacam-2025-07-09_01-13-22
[图片]

3. 场景集（hm3d）
obsutil sync obs://data-platform-shanghai/data/scene_datasets/cloudrobo_v1/train/suzhou-room-shengwei-metacam-2025-07-09_01-13-22 suzhou-room-shengwei-metacam-2025-07-09_01-13-22
[图片]

实现
streamvln habitat 环境安装
首先还是
# 初始化（会往 ~/.bashrc 写入 hook）
/opt/conda/bin/conda init bash  ||  ~/miniconda3/bin/conda init bash

# 重新进入一个登录 bash（让修改立刻生效）
exec bash -l

# 然后激活
conda create -n streamvln python=3.9
conda activate streamvln

apt install libegl1
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
# 如果不要用conda，就换为
#git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-sim.git \
# cd habitat-sim && pip install --no-cache-dir -r requirements.txt \
# python setup.py install --bullet --headless 
# 或者 pip install .[bullet,headless]
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines # install habitat_baselines
pip install -r requirements.txt # 整体

#  flash-attn安装：https://www.cnblogs.com/coldchair/p/18615384 # 会比较久
# https://github.com/InternRobotics/StreamVLN/issues/18
pip install packaging
pip install ninja
MAX_JOBS=4  pip install flash-attn==2.5.8 --no-build-isolation （会比较久）
如果不行就，换成了 :https://www.cnblogs.com/coldchair/p/18615384              attn_implementation="sdpa",，修改stremavln/args.py里面的指）
pip install protobuf==3.20.1
另外模型下载使用：https://gitcode.com/Winsleo/StreamVLN/commit/359cfcd3733561a6da90472008a95b603aa56170?ref=main
然后运行会报错
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "/root/miniconda3/envs/streamvln/lib/python3.9/site-packages/habitat_sim-0.2.4-py3.9-linux-x86_64.egg/habitat_sim/__init__.py", line 13, in <module>
import habitat_sim._ext.habitat_sim_bindings
ImportError: /lib/x86_64-linux-gnu/libOpenGL.so.0: undefined symbol: _glapi_tls_Current
验证库依赖关系是否正常 ：
# 检查 libOpenGL.so.0 是否链接了 libGLdispatch.so.0
ldd /lib/x86_64-linux-gnu/libOpenGL.so.0

# 检查 libGLdispatch.so.0 是否包含该符号
nm -D /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 | grep _glapi_tls_Current
如果 nm 命令没有输出，说明 libGLdispatch.so.0 版本过旧，需要更新。
发现确实没有，说明是libGLdispatch.so.0的版本太老了
apt update
apt reinstall libglvnd0 libglvnd-core-dev libgl1-mesa-glx

我们的实现主要做如下的事情：
1. 把hm3d objnav的任务数据转为stremavln的annotation格式
 python scripts/objnav_converters/objnav2r2r.py --input data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content/suzhou-room-shengwei-metacam-2025-07-09_01-13-22.json.gz --output data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/annotations.json
2. 在habiata里面跑通原本的hm3d+annotation获取图片的流程：https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data
  1. 内部自己实现是：https://github.com/kason-huang/StreamVLN/blob/master/scripts/objnav_converters/objnav2streamvln_1.py（这里用了重建转hm3d的场景会白沫（主要是用blender转换的），原因是：https://github.com/kason-huang/StreamVLN/blob/master/docs/objnav-3dgs-rendering-analysis.md
  2. 另外你需要去实现一个objnav_dataset来加载数据，主要是因为目前的数据里面的reference_action是自己新家的，具体的路径是 https://github.com/kason-huang/StreamVLN/commit/2ffc727e436220c7f44ec35fe65893e0398c5468#diff-458b5f2c3da055cf5ecd4043a936afb92111e703cae9e13e3901cc8bac1b741f
3. 加入高斯sensor
  1. https://github.com/kason-huang/StreamVLN/commit/2ffc727e436220c7f44ec35fe65893e0398c5468#diff-473f702a66377a8a0e61efd40cfb763449efb26ea1e1d30d3233842a5d2ffc21
  2. 配置环境
gs的安装方式：https://github.com/EmbodiedAILab/panoptic_gs/tree/b9f82cfa847f12f978e33a0718d1f2998a9f0132；
这里需要重新安装下torch 
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  3. 然后加入配置 https://github.com/kason-huang/StreamVLN/commit/25eda497edb0e50278c783548e73ac186473e675#diff-2fb83987086cc8d6d825d2278c1f73644ac1a345aeb7adc2b86219696ca72bcd
4. 生成图片
运行出图片的脚本
  conda activate streamvln-0226 
  PYTHONPATH=. python ./scripts/objnav_converters/objnav2streamvln.py
5. 转为lerobot 
conda activate lerobot-transfer
python scripts/dataset_converters/r2r2lerobot.py --data_dir "./data/trajectory_data" --output_dir "./data/lerobot-shengwei-reconstruction" --dataset_name "objectnav/cloudrobo_v1_l3mvn" --repo_id "cloudrobo/lerobot_shengwei_reconstruction" --fps 3 --start_idx 0 --end_idx 200 --overwrite





---
转化流程
1. Cloudrobo objnav -》streamvln annotation
 python scripts/objnav_converters/objnav2r2r.py --input data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content/suzhou-room-shengwei-metacam-2025-07-09_01-13-22.json.gz --output data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/annotations.json
2. streamvln annotation generate image
  now, try to implement the process of generate the image just like @scripts/objnav_converters/objnav2streamvln.py for
    @data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content/suzhou-room-shengwei-metacam-2025-07-09_01-13-22.json.gz,
    the scene_path is @data/scene_datasets/cloudrobo_v1/train/suzhou-room-shengwei-metacam-2025-07-09_01-13-22; for the analyze of the data
    structure of suzhou-room-shengwei-metacam-2025-07-09_01-13-22.json.gz you can read the  @docs/objnav_data_structure.md, you can use
    the AskUserQuestion to disucss with me for more detail, do no write the code directly

需要扩展下habitat_extensions里面的dtaset，需要用https://github.com/EmbodiedAILab/ovon/blob/master/ovon/dataset/objectnav_dataset.py#L233 config的dataset的type是 ObjectNav-v1
如果参考张博那边的话啊，就是用的配置文件：https://github.com/ZBoIsHere/NavDataGeneration/blob/master/NavTrajSampleGeneration/L3MVN/envs/habitat/configs/tasks/objectnav_hm3d.yaml，然后剩下的自己构建
之前诗晴高的是：https://github.com/EmbodiedAILab/ovon/blob/master/config/experiments/transformer_il_3dgs.yaml
要注意角度与机器人高度什么的
gs的安装方式：https://github.com/EmbodiedAILab/panoptic_gs/tree/b9f82cfa847f12f978e33a0718d1f2998a9f0132；
这里需要重新安装下torch 
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
运行出图片的脚本
  conda activate streamvln-0226 
  PYTHONPATH=. python ./scripts/objnav_converters/objnav2streamvln.py
3. Streamvln annotation + image -> lerobot

conda activate lerobot-transfer
python scripts/dataset_converters/r2r2lerobot.py --data_dir "./data/trajectory_data" --output_dir "./data/lerobot-shengwei-reconstruction" --dataset_name "objectnav/cloudrobo_v1_l3mvn" --repo_id "cloudrobo/lerobot_shengwei_reconstruction" --fps 3 --start_idx 0 --end_idx 200 --overwrite
